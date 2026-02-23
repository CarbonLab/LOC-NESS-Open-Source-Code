import os
import requests
import pandas as pd
from io import StringIO
from bs4 import BeautifulSoup, Tag
import re
import gc

class GliderDataLoader:
    def __init__(self, filenames=None, sample_rate=None, include_qc=False, range_start=None, range_end=None):
        """
        Initialize DataLoader. Always fetches available RT.txt files.

        Parameters:
            filenames (str or list of str, optional): Files to load. If None, defaults to most recent.
        """
        self.folder_url = "https://www3.mbari.org/lobo/Data/GliderVizData/"
        self.available_files = self._get_available_files()
        self.sample_rate = sample_rate
        self.include_qc = include_qc
        self.range_start = range_start
        self.range_end = range_end
        if not self.available_files:
            raise FileNotFoundError("No RT.txt files found at the specified URL.")

        if filenames is None:
            self.file_list = [self.available_files[-1]]  # default to most recent
        elif isinstance(filenames, str):
            self.file_list = [filenames]
        else:
            self.file_list = filenames

    def _get_available_files(self):
        """Scrape and return sorted list of available RT.txt files from the folder."""
        response = requests.get(self.folder_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        files = [
            os.path.basename(str(a['href']))
            for a in soup.find_all('a', href=True)
            if isinstance(a, Tag) and 'RT.txt' in str(a['href'])
        ]
        return sorted(files)

    def get_available_files(self):
        """Public method to get list of available files."""
        return self.available_files

    def get_latest_file(self):
        """Return the most recent available RT.txt file."""
        return self.available_files[-1]

    def load_data(self):
        """
        Load and clean all initialized files.

        Returns:
            A single concatenated DataFrame from all files.
        """
        dfs = []  # Fresh list each time

        for fname in self.file_list:
            file_url = self.folder_url + fname
            try:
                file_response = requests.get(file_url)
                file_content = StringIO(file_response.text)
                df = pd.read_csv(file_content, delimiter="\t", skiprows=6)
            except Exception as e:
                continue

            # Clean and process
            df.columns = df.columns.str.replace('Â', '', regex=False)
            df.replace([-1e10, -999], pd.NA, inplace=True)
            df['Date'] = pd.to_datetime(df['mon/day/yr'], format='%m/%d/%Y', errors='coerce')
            df['Datetime'] = pd.to_datetime(
                df['mon/day/yr'] + ' ' + df['hh:mm'],
                format='%m/%d/%Y %H:%M',
                errors='coerce'
            )
            df['unixTimestamp'] = df['Datetime'].astype('int64') // 10**9
            if not self.include_qc:
                # Drop columns that contain "_QC" to save memory
                qc_columns = [col for col in df.columns if '_QC' in col]
                df.drop(columns=qc_columns, inplace=True)
            if self.range_start is not None and self.range_end is not None:
                df = df[(df['unixTimestamp'] >= self.range_start) & (df['unixTimestamp'] <= self.range_end)]
            if 'PHIN_CANYONB[Total]' in df.columns and 'pHinsitu[Total]' in df.columns:
                df['pHin_Canb_Delta'] = df['pHinsitu[Total]'] - df['PHIN_CANYONB[Total]']
            else:
                df['pHin_Canb_Delta'] = pd.NA

            df['source_file'] = fname
            dfs.append(df)

        # Ensure all DataFrames have the same columns
        if dfs:
            all_columns = set()
            for df in dfs:
                all_columns.update(df.columns)

            # Add missing columns
            for i, df in enumerate(dfs):
                missing_columns = all_columns - set(df.columns)
                for col in missing_columns:
                    if 'Date' in col or 'Datetime' in col:
                        dfs[i][col] = pd.Series(dtype='datetime64[ns]')
                    elif 'unixTimestamp' in col:
                        dfs[i][col] = pd.Series(dtype='int64')
                    elif any(numeric_col in col for numeric_col in ['Temp', 'Sal', 'O2', 'pH', 'Depth', 'Lat', 'Lon']):
                        dfs[i][col] = pd.Series(dtype='float64')
                    else:
                        dfs[i][col] = pd.Series(dtype='object')

            # Ensure same column order
            column_order = sorted(list(all_columns))
            for i, df in enumerate(dfs):
                dfs[i] = df.reindex(columns=column_order)

            # Concatenate and resample
            df_combined = pd.concat(dfs, ignore_index=True)

            # # Clear the list to free memory
            # dfs.clear()  # ADD THIS LINE!
            # Explicitly free intermediates
            del dfs
            gc.collect()

            # Apply resampling if specified
            if self.sample_rate is not None and self.sample_rate > 1:
                df_resampled = df_combined.iloc[::self.sample_rate].copy()
                return df_resampled
            else:
                return df_combined
        else:
            return pd.DataFrame()  # Return empty DataFrame if no files

class GulfStreamLoader:
    def __init__(self):
        """
        Initialize DataLoader.
        """
        self.url = "https://ocean.weather.gov/gulf_stream_latest.txt"  # Fixed: removed extra quotes

    def load_data(self):
        response = requests.get(self.url)
        data = response.text

        # Step 2: Locate the coordinates section
        match = re.search(r'RMKS/1\. GULF STREAM NORTH WALL DATA FOR.*?:\s*(.*?)(?:RMKS/|$)', data, re.S)
        if not match:
            raise ValueError('Could not locate the Gulf Stream coordinates section.')
        coordinates_text = match.group(1).strip()

        # Step 3: Extract coordinate pairs (format like 25.5N80.1W)
        coordinate_strings = re.findall(r'(\d+\.\d+N\d+\.\d+W)', coordinates_text)

        # Step 4: Convert to decimal degrees and store in lists
        latitudes = []
        longitudes = []

        for pair in coordinate_strings:
            match_coords = re.match(r'(\d+\.\d+)N(\d+\.\d+)W', pair)
            if match_coords:  # Added error checking
                lat_str, lon_str = match_coords.groups()
                latitudes.append(float(lat_str))
                longitudes.append(-float(lon_str))  # W longitude is negative

        # Step 5: Create a pandas DataFrame
        gulfstreamcoords = pd.DataFrame({
            'Lat': latitudes,
            'Lon': longitudes
        })
        return gulfstreamcoords
class MapDataLoader:
    def __init__(self, filenames=None):
        """
        Initialize DataLoader. Always fetches available sensor-data.txt files.

        Parameters:
            filenames (str or list of str, optional): Files to load. If None, defaults to most recent.
        """
        self.folder_url = "https://www3.mbari.org/lobo/Data/GliderVizData/"
        self.available_files = self._get_available_files()

        if not self.available_files:
            raise FileNotFoundError("No LocNessMapProduct files found at the specified URL.")

        if filenames is None:
            self.file_list = [self.available_files[-1]]  # default to most recent
        elif isinstance(filenames, str):
            self.file_list = [filenames]
        else:
            self.file_list = filenames

    def _get_available_files(self):
        """Scrape and return sorted list of available RT.txt files from the folder."""
        response = requests.get(self.folder_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        files = [
            os.path.basename(str(a['href']))
            for a in soup.find_all('a', href=True)
            if isinstance(a, Tag) and 'LocnessMapProduct.txt' in str(a['href'])
        ]
        return sorted(files)

    def get_available_files(self):
        """Public method to get list of available files."""
        return self.available_files

    def get_latest_file(self):
        """Return the most recent available sensor-data.txt file."""
        return self.available_files[-1]

    def load_data(self):
        """
        Load and clean all initialized files.

        Returns:
            A single concatenated DataFrame from all files.
        """
        file_url = self.folder_url + self.file_list[0]
        file_response = requests.get(file_url)
        file_content = StringIO(file_response.text)
        df = pd.read_csv(file_content, delimiter=",", dtype={'Cruise': 'category', 'Platform': 'category', 'Layer': 'category', 'CastDirection': 'category'})

        # Clean
        df.columns = df.columns.str.replace('Â', '', regex=False)
        df.replace([-1e10, -999], pd.NA, inplace=True)
        df['Datetime'] = pd.to_datetime(df['unixTimestamp'], unit='s', errors='coerce')

        # Remove rows where lat or lon are missing
        df = df.dropna(subset=['lat', 'lon'])

        return df

class GliderGridDataLoader:
    def __init__(self, filenames=None):
        """
        Initialize DataLoader. Always fetches available sensor-data.txt files.

        Parameters:
            filenames (str or list of str, optional): Files to load. If None, defaults to most recent.
        """
        self.folder_url = "https://www3.mbari.org/lobo/Data/GliderVizData/"
        self.available_files = self._get_available_files()

        if not self.available_files:
            raise FileNotFoundError("No grid.txt files found at the specified URL.")

        if filenames is None:
            self.file_list = [self.available_files[-1]]  # default to most recent
        elif isinstance(filenames, str):
            self.file_list = [filenames]
        else:
            self.file_list = filenames

    def _get_available_files(self):
        """Scrape and return sorted list of available RT.txt files from the folder."""
        response = requests.get(self.folder_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        files = [
            os.path.basename(str(a['href']))
            for a in soup.find_all('a', href=True)
            if isinstance(a, Tag) and 'grid_lat_lon' in str(a['href'])
        ]
        return sorted(files)

    def get_available_files(self):
        """Public method to get list of available files."""
        return self.available_files

    def get_latest_file(self):
        """Return the most recent available sensor-data.txt file."""
        return self.available_files[-1]

    def load_data(self):
        """
        Load and clean all initialized files.

        Returns:
            A single concatenated DataFrame from all files.
        """
        file_url = self.folder_url + self.file_list[0]
        file_response = requests.get(file_url)
        file_content = StringIO(file_response.text)
        df = pd.read_csv(file_content, delimiter=",", dtype={'Grid_ID': 'string', 'Lat': 'float64', 'Lon': 'float64'})

        # Clean
        df.columns = df.columns.str.replace('Â', '', regex=False)

        return df

class MPADataLoader:
    def __init__(self, filenames=None):
        """
        Initialize DataLoader. Always fetches available sensor-data.txt files.

        Parameters:
            filenames (str or list of str, optional): Files to load. If None, defaults to most recent.
        """
        self.folder_url = "https://www3.mbari.org/lobo/Data/GliderVizData/"
        self.available_files = self._get_available_files()

        if not self.available_files:
            raise FileNotFoundError("Stellwagon.txt files found at the specified URL.")

        if filenames is None:
            self.file_list = [self.available_files[-1]]  # default to most recent
        elif isinstance(filenames, str):
            self.file_list = [filenames]
        else:
            self.file_list = filenames

    def _get_available_files(self):
        """Scrape and return sorted list of available RT.txt files from the folder."""
        response = requests.get(self.folder_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        files = [
            os.path.basename(str(a['href']))
            for a in soup.find_all('a', href=True)
            if isinstance(a, Tag) and 'Stellwagen_Bank_National_Marine_Sanctuary' in str(a['href'])
        ]
        return sorted(files)

    def get_available_files(self):
        """Public method to get list of available files."""
        return self.available_files

    def get_latest_file(self):
        """Return the most recent available sensor-data.txt file."""
        return self.available_files[-1]

    def load_data(self):
        """
        Load and clean all initialized files.

        Returns:
            A single concatenated DataFrame from all files.
        """
        file_url = self.folder_url + self.file_list[0]
        file_response = requests.get(file_url)
        file_content = StringIO(file_response.text)
        df = pd.read_csv(file_content, delimiter="\t", dtype={'Grid_ID': 'string', 'Lat': 'float64', 'Lon': 'float64'})

        # Clean
        df.columns = df.columns.str.replace('Â', '', regex=False)

        return df

class gomofsdataloader:
    def __init__(self, filenames=None):
        """
        Initialize DataLoader. Always fetches availabledata.txt files.

        Parameters:
            filenames (str or list of str, optional): Files to load. If None, defaults to most recent.
        """
        self.folder_url = "https://www3.mbari.org/lobo/Data/GliderVizData/"
        self.available_files = self._get_available_files()

        if not self.available_files:
            raise FileNotFoundError("gomofs.txt files found at the specified URL.")

        if filenames is None:
            self.file_list = [self.available_files[-1]]  # default to most recent
        elif isinstance(filenames, str):
            self.file_list = [filenames]
        else:
            self.file_list = filenames

    def _get_available_files(self):
        """Scrape and return sorted list of available RT.txt files from the folder."""
        response = requests.get(self.folder_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        files = [
            os.path.basename(str(a['href']))
            for a in soup.find_all('a', href=True)
            if isinstance(a, Tag) and 'current_gomofs.txt' in str(a['href'])
        ]
        return sorted(files)

    def get_available_files(self):
        """Public method to get list of available files."""
        return self.available_files

    def get_latest_file(self):
        """Return the most recent available sensor-data.txt file."""
        return self.available_files[-1]

    def load_data(self):
        """
        Load and clean all initialized files.

        Returns:
            A single concatenated DataFrame from all files.
        """
        file_url = self.folder_url + self.file_list[0]
        file_response = requests.get(file_url)
        file_content = StringIO(file_response.text)
        df = pd.read_csv(file_content, delimiter=",", dtype={'trajectory': 'float64', 'obs': 'float64' ,'lat': 'float64', 'lon': 'float64', 'z': 'float64'}, parse_dates=['time'])
        # Clean
        df.columns = df.columns.str.replace('Â', '', regex=False)

        return df

class doppiodataloader:
    def __init__(self, filenames=None):
        """
        Initialize DataLoader. Always fetches availabledata.txt files.

        Parameters:
            filenames (str or list of str, optional): Files to load. If None, defaults to most recent.
        """
        self.folder_url = "https://www3.mbari.org/lobo/Data/GliderVizData/"
        self.available_files = self._get_available_files()

        if not self.available_files:
            raise FileNotFoundError("gomofs.txt files found at the specified URL.")

        if filenames is None:
            self.file_list = [self.available_files[-1]]  # default to most recent
        elif isinstance(filenames, str):
            self.file_list = [filenames]
        else:
            self.file_list = filenames

    def _get_available_files(self):
        """Scrape and return sorted list of available RT.txt files from the folder."""
        response = requests.get(self.folder_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        files = [
            os.path.basename(str(a['href']))
            for a in soup.find_all('a', href=True)
            if isinstance(a, Tag) and 'current_doppio.txt' in str(a['href'])
        ]
        return sorted(files)

    def get_available_files(self):
        """Public method to get list of available files."""
        return self.available_files

    def get_latest_file(self):
        """Return the most recent available sensor-data.txt file."""
        return self.available_files[-1]

    def load_data(self):
        """
        Load and clean all initialized files.

        Returns:
            A single concatenated DataFrame from all files.
        """
        file_url = self.folder_url + self.file_list[0]
        file_response = requests.get(file_url)
        file_content = StringIO(file_response.text)
        df = pd.read_csv(file_content, delimiter=",", dtype={'trajectory': 'float64', 'obs': 'float64' ,'lat': 'float64', 'lon': 'float64', 'z': 'float64'}, parse_dates=['time'])
        # Clean
        df.columns = df.columns.str.replace('Â', '', regex=False)

        return df