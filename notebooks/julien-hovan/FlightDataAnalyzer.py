import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

class FlightDataAnalyzer:
    def __init__(self, df):
        self.df = df.copy()
        # Convert timestamp columns to datetime
        time_columns = ['firstseen', 'lastseen', 'day']
        for col in time_columns:
            self.df[col] = pd.to_datetime(self.df[col])
        
        # Calculate flight duration in minutes
        self.df['flight_duration'] = (self.df['lastseen'] - self.df['firstseen']).dt.total_seconds() / 60

    def analyze_flight_durations(self):
        """Analyze flight durations statistics"""
        stats = {
            'mean_duration': self.df['flight_duration'].mean(),
            'median_duration': self.df['flight_duration'].median(),
            'min_duration': self.df['flight_duration'].min(),
            'max_duration': self.df['flight_duration'].max()
        }
        return stats

    def analyze_hourly_distribution(self):
        """Analyze hourly distribution of flights"""
        self.df['hour'] = self.df['firstseen'].dt.hour
        hourly_flights = self.df['hour'].value_counts().sort_index()
        return hourly_flights
    
    def analyze_different_airport_routes(self):
        """
        Analyze most common routes where departure and arrival airports are different.
        Excludes routes where departure or arrival information is missing.
        
        Returns:
            pd.Series: Top 10 routes with different departure and arrival airports
        """
        # Filter for routes where departure != arrival and both are not null
        valid_routes = self.df[
            (self.df['departure'].notna()) & 
            (self.df['arrival'].notna()) & 
            (self.df['departure'] != self.df['arrival'])
        ]
        
        # Group and count the routes
        routes = valid_routes.groupby(['departure', 'arrival']).size().sort_values(ascending=False)
        
        return routes.head(10)

    def analyze_routes(self):
        """Analyze most common routes"""
        routes = self.df.groupby(['departure', 'arrival']).size().sort_values(ascending=False)
        return routes.head(10)

    def analyze_airlines(self):
        """Analyze most active airlines based on callsigns"""
        airline_freq = self.df['callsign'].value_counts()
        return airline_freq.head(10)

    def plot_hourly_distribution(self):
        """Plot hourly distribution of flights"""
        plt.figure(figsize=(12, 6))
        hourly = self.analyze_hourly_distribution()
        ax = hourly.plot(kind='bar')
        plt.title('Hourly Distribution of Flights')
        plt.xlabel('Hour of Day')
        plt.ylabel('Number of Flights')
        
        # Ensure x-axis ticks are integers (hours)
        ax.set_xticks(range(24))
        ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45, ha='right')
        
        plt.tight_layout()
        return plt

    def plot_flight_duration_histogram(self):
        """Plot histogram of flight durations"""
        plt.figure(figsize=(12, 6))
        plt.hist(self.df['flight_duration'].dropna(), bins=50)
        plt.title('Distribution of Flight Durations')
        plt.xlabel('Duration (minutes)')
        plt.ylabel('Number of Flights')
        plt.tight_layout()
        return plt

    def analyze_missing_data(self):
        """Analyze percentage of missing data in each column"""
        missing_data = (self.df.isna().sum() / len(self.df) * 100).round(2)
        return missing_data

    def analyze_seasonal_patterns(self, freq='M'):
        """
        Analyze seasonal patterns in flight frequency
        
        Args:
            freq (str): Frequency for grouping. Options:
                'M': Monthly
                'W': Weekly
                'D': Daily
                'Q': Quarterly
        
        Returns:
            pd.Series: Flight counts grouped by the specified frequency
        """
        # Extract the date from firstseen
        self.df['date'] = self.df['firstseen'].dt.date
        
        # Create time-based grouping
        if freq == 'M':
            return self.df.groupby(self.df['firstseen'].dt.to_period('M')).size()
        elif freq == 'W':
            return self.df.groupby(self.df['firstseen'].dt.isocalendar().week).size()
        elif freq == 'Q':
            return self.df.groupby(self.df['firstseen'].dt.to_period('Q')).size()
        else:  # Daily is default
            return self.df.groupby('date').size()

    def plot_seasonal_patterns(self, freq='M'):
        """
        Plot seasonal patterns in flight frequency
        
        Args:
            freq (str): Frequency for grouping (M, W, D, Q)
        """
        seasonal_data = self.analyze_seasonal_patterns(freq)
        
        plt.figure(figsize=(15, 6))
        ax = seasonal_data.plot(kind='line', marker='o')
        
        # Customize plot based on frequency
        freq_labels = {
            'M': 'Monthly',
            'W': 'Weekly',
            'D': 'Daily',
            'Q': 'Quarterly'
        }
        
        plt.title(f'{freq_labels[freq]} Distribution of Flights')
        plt.xlabel(f'{freq_labels[freq]} Period')
        plt.ylabel('Number of Flights')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        return plt

    def analyze_seasonal_routes(self, season_type='month'):
        """
        Analyze most common routes by season
        
        Args:
            season_type (str): Type of seasonal grouping:
                'month': Monthly analysis
                'quarter': Quarterly analysis
                'season': Traditional seasons (Spring, Summer, Fall, Winter)
        
        Returns:
            dict: Dictionary containing top routes for each season
        """
        def get_season(month):
            if month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Spring'
            elif month in [6, 7, 8]:
                return 'Summer'
            else:
                return 'Fall'

        if season_type == 'month':
            self.df['season'] = self.df['firstseen'].dt.month
        elif season_type == 'quarter':
            self.df['season'] = self.df['firstseen'].dt.quarter
        else:  # Traditional seasons
            self.df['season'] = self.df['firstseen'].dt.month.map(get_season)

        seasonal_routes = {}
        for season in self.df['season'].unique():
            season_data = self.df[self.df['season'] == season]
            routes = season_data.groupby(['departure', 'arrival']).size().sort_values(ascending=False)
            seasonal_routes[season] = routes.head(5)  # Top 5 routes per season

        return seasonal_routes
