import pandas as pd
import plotly.graph_objs as go
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px
import statsmodels.api as sm
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import adfuller, kpss


class DataCleaning():
    def __init__(self, data_path, building_name, energy_type, z_threshold):
        self.data_path = data_path
        self.building_name = building_name
        self.data = None
        self.z_threshold = z_threshold
        self.energy_type = energy_type

    def clean_data(self):
        """
        Main method to perform data cleaning and data analysis
        """
        self.data = pd.read_csv(self.data_path)
        self.data
        self.set_data_index()
        self.missing_values()
        self.descriptive_stats()
        self.remove_outliers()
        self.seasonal_decomposition()
        self.cyclical_features()
        self.correlation_analysis()
        return self.data
    
    def set_data_index(self):
        """
        A method to create an index for each time series and clean the data format
        """
        if self.energy_type == "Fjv":
            self.data['Period'] = pd.to_datetime(self.data['Period'])
            self.data.set_index('Period', inplace=True)

            # Create a trace for the time series
            trace = go.Scatter(x=self.data.index, y=self.data['corrected_value'], mode='lines', name='Corrected Value')

            # Create layout for the plot
            layout = go.Layout(title=f'District Heating Consumption {self.building_name}',
                            xaxis=dict(title='Date'),
                            yaxis=dict(title='Consumption'))

            # Create Figure object
            fig = go.Figure(data=[trace], layout=layout)

            # Show the plot
            fig.show()
        
        if self.energy_type == "EL":
            # Set Datetime as index
            self.data = self.data[~self.data['Förbrukning'].str.contains('Förbrukning')]

            self.data['Period'] = pd.to_datetime(self.data['Period'])

            # Sort the DataFrame by the date column
            self.data = self.data.sort_values(by='Period')

            self.data['Förbrukning'] = pd.to_numeric(self.data['Förbrukning'])

            self.data.set_index('Period', inplace=True)

            # Rename Index name
            self.data.index.name = 'Period'

            # Create a trace for the time series
            trace = go.Scatter(x=self.data.index, y=self.data['Förbrukning'], mode='lines', name='Consumption')

            # Create layout for the plot
            layout = go.Layout(title=f'Electricity Consumption {self.building_name}',
                            xaxis=dict(title='Date'),
                            yaxis=dict(title='Consumption'))

            # Create Figure object
            fig = go.Figure(data=[trace], layout=layout)

            # Show the plot
            fig.show()

        if self.energy_type == "EL_Kinder":
            # Define the Swedish to English month mapping
            swedish_to_english_months = {
                'jan': 'Jan', 'feb': 'Feb', 'mar': 'Mar', 'apr': 'Apr', 'maj': 'May', 'jun': 'Jun',
                'jul': 'Jul', 'aug': 'Aug', 'sep': 'Sep', 'okt': 'Oct', 'nov': 'Nov', 'dec': 'Dec'
            }

            # Define a function to unify time format
            def unify_time_format(date_str):
                if date_str == 'Period':
                    return pd.NaT

                try:
                    if "T" in date_str and "Z" in date_str:
                        return pd.to_datetime(date_str, errors='coerce', utc=True).tz_convert(None)
                except Exception:
                    pass

                date_str_lower = date_str.lower()
                for swe, eng in swedish_to_english_months.items():
                    date_str_lower = date_str_lower.replace(swe, eng.capitalize())

                date_formats = ["%d %b %Y %H:%M", "%d/%m/%y %H:%M", "%d/%m/%Y %H:%M"]
                for fmt in date_formats:
                    try:
                        parsed_date = pd.to_datetime(date_str_lower, format=fmt, exact=True, errors='coerce')
                        if pd.notnull(parsed_date):
                            return parsed_date
                    except ValueError:
                        continue

                return pd.NaT

            # Apply the function to the 'Period' column
            self.data['Period'] = self.data['Period'].astype(str).apply(unify_time_format)

            # Drop rows with invalid dates or missing 'Förbrukning'
            self.data.dropna(subset=['Period', 'Förbrukning'], inplace=True)

            # Convert 'Förbrukning' to a numeric type, handling potential decimal separators
            self.data['Förbrukning'] = pd.to_numeric(self.data['Förbrukning'].str.replace(',', '.'), errors='coerce')

            # Convert 'Period' column to datetime
            self.data['Period'] = pd.to_datetime(self.data['Period'], errors='coerce')

            # Sort and reset the index
            self.data.sort_values(by='Period', inplace=True)
            self.data.reset_index(drop=True, inplace=True)

            self.data.index.name = 'Period'
            self.data.set_index('Period', inplace=True)

            # Convert the datetime index to UTC timezone
            self.data.index = self.data.index.tz_localize(None).tz_localize('UTC')

            # Create a trace for the time series
            trace = go.Scatter(x=self.data.index, y=self.data['Förbrukning'], mode='lines', name='Consumption')

            # Create layout for the plot
            layout = go.Layout(title='District Heating Consumption Tallbacka',
                            xaxis=dict(title='Date'),
                            yaxis=dict(title='Consumption'))

            # Create Figure object
            fig = go.Figure(data=[trace], layout=layout)

            # Show the plot
            fig.show()


    def missing_values(self):
        """
        A method to encounter missing values and perform linear interpolation in case of missing values
        """
        missing_values = self.data.isnull().sum()
        print(f'Missing values: {missing_values}')
        if self.energy_type == "Fjv":
            # Interpolate missing values
            self.data = self.data.infer_objects()
            self.data = self.data.interpolate(method='linear')

    def descriptive_stats(self):
        """
        A method to perform descriptive statistics for the time series
        """
        if self.energy_type == "Fjv":
            # Plot histogram with Plotly
            fig = px.histogram(self.data, x='corrected_value', nbins=30, title='Distribution of corrected_value')
            fig.update_layout(xaxis_title='Value', yaxis_title='Frequency')
            fig.show()

            # Calculate kurtosis and skewness
            kurtosis_value = stats.kurtosis(self.data['corrected_value'])
            skewness_value = stats.skew(self.data['corrected_value'])
            print("Kurtosis:", kurtosis_value)
            print("Skewness:", skewness_value)

            # Calculate descriptive statistics
            mean_value = self.data['corrected_value'].mean()
            median_value = self.data['corrected_value'].median()
            mode_value = self.data['corrected_value'].mode()[0]  
            std_dev = self.data['corrected_value'].std()

            # Print descriptive statistics
            print("Mean:", mean_value)
            print("Median:", median_value)
            print("Mode:", mode_value)
            print("Standard Deviation:", std_dev)

        if self.energy_type == "EL" or self.energy_type == "EL_Kinder":
            
            fig = px.histogram(self.data, x='Förbrukning', nbins=30, title='Distribution of Förbrukning')
            fig.update_layout(xaxis_title='Value', yaxis_title='Frequency')
            fig.show()

            # Calculate kurtosis and skewness
            kurtosis_value = stats.kurtosis(self.data['Förbrukning'])
            skewness_value = stats.skew(self.data['Förbrukning'])

            print("Kurtosis:", kurtosis_value)
            print("Skewness:", skewness_value)

            # Assuming 'df' is your DataFrame containing the data
            # Calculate descriptive statistics
            mean_value = self.data['Förbrukning'].mean()
            median_value = self.data['Förbrukning'].median()
            mode_value = self.data['Förbrukning'].mode()[0]  
            std_dev = self.data['Förbrukning'].std()

            # Print descriptive statistics
            print("Mean:", mean_value)
            print("Median:", median_value)
            print("Mode:", mode_value)
            print("Standard Deviation:", std_dev)

    def remove_outliers(self):
        """
        A method to remove outliers based on a threshold using Z-score
        """
        if self.energy_type == "Fjv":
            # Identifying rows with 0 values in 'corrected_value' feature
            zero_values = self.data[self.data['corrected_value'] == 0]

            # Remove rows with 0 values in 'corrected_value' feature
            self.data = self.data[self.data['corrected_value'] != 0]

            # Calculate Z-scores for the data
            self.data['z_score'] = (self.data['corrected_value'] - self.data['corrected_value'].mean()) / self.data['corrected_value'].std()

            # Identify outliers based on Z-score
            outliers = self.data[abs(self.data['z_score']) > self.z_threshold]

            # Print the number of outliers excluding zeros
            print("Number of outliers (excluding zeros):", outliers.shape[0])

            # Plot original data
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=self.data.index, y=self.data['corrected_value'], mode='lines'))

            # Add non-zero outliers to the plot
            fig.add_trace(go.Scatter(x=outliers.index, y=outliers['corrected_value'], mode='markers', 
                                    marker=dict(color='red', size=8), name='Outliers (Z-Score)'))

            # Add a trace for zero value data points as green dots
            if not zero_values.empty:
                fig.add_trace(go.Scatter(x=zero_values.index, y=zero_values['corrected_value'], mode='markers', 
                                        marker=dict(color='green', size=8, symbol='circle'), name='Zero Values (Outliers)'))

            # Update layout
            fig.update_layout(title='Time Series with Outliers (Z-Score and Zero Values)',
                            xaxis_title='Date',
                            yaxis_title='Corrected Value')

            # Show plot
            fig.show()

            # Remove non-zero outliers based on the Z-score
            self.data = self.data[abs(self.data['z_score']) <= self.z_threshold]

        if self.energy_type == "EL" or self.energy_type == "EL_Kinder":

            # Remove rows with 0 values in 'corrected_value' feature
            self.data = self.data[self.data['Förbrukning'] != 0]

            # Calculate Z-scores for the data
            self.data['z_score'] = (self.data['Förbrukning'] - self.data['Förbrukning'].mean()) / self.data['Förbrukning'].std()

            # Identify outliers based on Z-score
            outliers = self.data[abs(self.data['z_score']) > z_threshold]

            # Print the number of outliers excluding zeros
            print("Number of outliers (excluding zeros):", outliers.shape[0])

            # Plot original data
            fig = go.Figure()
            # Create a trace for the time series
            fig.add_trace(go.Scatter(x=self.data.index, y=self.data['Förbrukning'], mode='lines', name='Consumption'))

            # Add non-zero outliers to the plot
            fig.add_trace(go.Scatter(x=outliers.index, y=outliers['Förbrukning'], mode='markers', 
                                    marker=dict(color='red', size=8), name='Outliers (Z-Score)'))

            # Update layout
            fig.update_layout(title='Time Series with Outliers (Z-Score and Zero Values)',
                            xaxis_title='Date',
                            yaxis_title='Consumption')

            # Show plot
            fig.show()

            # Remove outliers based on the Z-score
            self.data = self.data[abs(self.data['z_score']) <= z_threshold]

    def seasonal_decomposition(self):
        """
        A method to plot a seasonal decomposition and to calculate the stationarity of the time series
        """
        if self.energy_type == "Fjv":
            index_name = "corrected_value"
        if self.energy_type == "EL" or self.energy_type == "EL_Kinder":
            index_name = "Förbrukning"
        # Decompose data by selecting the appropriate frequency
        decomp = sm.tsa.seasonal_decompose(self.data[index_name], period=365)

        # Create subplots
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                            subplot_titles=("Original Data", "Trend", "Seasonal", "Residual"))

        # Add original data trace
        fig.add_trace(go.Scatter(x=self.data.index, y=self.data[index_name], mode='lines', name='Original'),
                    row=1, col=1)

        # Add trend trace
        fig.add_trace(go.Scatter(x=self.data.index, y=decomp.trend, mode='lines', name='Trend'),
                    row=2, col=1)

        # Add seasonal trace
        fig.add_trace(go.Scatter(x=self.data.index, y=decomp.seasonal, mode='lines', name='Seasonal'),
                    row=3, col=1)

        # Add residual trace
        fig.add_trace(go.Scatter(x=self.data.index, y=decomp.resid, mode='lines', name='Residual'),
                    row=4, col=1)

        # Update layout
        fig.update_layout(title=f'Seasonal Decomposition {self.building_name}',
                        xaxis_title="Date",
                        showlegend=False)

        # Show plot
        fig.show()

        # Stationarity test using Augmented Dickey-Fuller test
        adf_result = adfuller(self.data[index_name].dropna())

        print('ADF Statistic: %f' % adf_result[0])
        print('p-value: %f' % adf_result[1])
        print('Critical Values:')
        for key, value in adf_result[4].items():
            print('\t%s: %.3f' % (key, value))


    def cyclical_features(self):
        """
        A method to create cyclical features for sin and cos of year, month and week day.
        """
        # Extract day of week, day of month, month, and day of year from the date
        self.data['dayofweek'] = self.data.index.dayofweek
        self.data['day'] =self.data.index.day
        self.data['month'] = self.data.index.month
        self.data['dayofyear'] = self.data.index.dayofyear

        # Create cyclical features for day of week
        self.data['dayofweek_sin'] = np.sin(2 * np.pi * self.data['dayofweek'] / 7)
        self.data['dayofweek_cos'] = np.cos(2 * np.pi * self.data['dayofweek'] / 7)

        # Create cyclical features for day of month
        self.data['day_sin'] = np.sin(2 * np.pi * self.data['day'] / 31)
        self.data['day_cos'] = np.cos(2 * np.pi * self.data['day'] / 31)

        # Create cyclical features for month
        self.data['month_sin'] = np.sin(2 * np.pi * self.data['month'] / 12)
        self.data['month_cos'] = np.cos(2 * np.pi * self.data['month'] / 12)

        # Create cyclical features for day of year
        self.data['dayofyear_sin'] = np.sin(2 * np.pi * self.data['dayofyear'] / 365)
        self.data['dayofyear_cos'] = np.cos(2 * np.pi * self.data['dayofyear'] / 365)
        
    def correlation_analysis(self):
        """
        A method to perform corerlation analysis with Pearson method
        """
        if self.energy_type == "Fjv":
            index_name = "corrected_value"
            features   = ['corrected_value', 'Akt', 'dayofweek_sin', 'dayofweek_cos','month_sin','month_cos','dayofyear_sin','dayofyear_cos','day_sin','day_cos']
        if self.energy_type == "EL" or self.energy_type == "EL_Kinder":
            index_name = "Förbrukning"
            features   = ['Förbrukning', 'dayofweek_sin', 'dayofweek_cos','month_sin','month_cos','dayofyear_sin','dayofyear_cos','day_sin','day_cos']

        # Calculate correlation coefficients
        corr_matrix = self.data[features].corr()
        display(corr_matrix[index_name])

        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(11, 9))

        # Generate a custom diverging colormap
        cmap = sns.color_palette("viridis", as_cmap=True) 

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt=".2f")  # Adjusting vmin and vmax

        plt.show()