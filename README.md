# Fat-tailed Distributions and Extreme Events Project

This project investigates financial data to analyze fat tails in market returns. It utilizes historical data to calculate returns and generate probability distributions for observing extreme market events. The project offers a comparison between the risk of these extreme events and a standard Gaussian distribution. The findings shed light on the importance of considering fat tails in financial modeling for improved risk comprehension and management.

## Instructions

1. **Install necessary packages**: This project uses the `yfinance` Python package to interface with Yahoo Finance's API for gathering historical data on various financial instruments.

2. **Define constants**: Set the ticker, start date, and end date for the data you want to gather.

3. **Gather Data**: Utilize the `yfinance` package to download historical Bitcoin price data in daily, weekly, monthly, and quarterly time frames.

4. **Calculate Rate of Returns**: Use logarithmic returns to calculate the rate of return between two subsequent prices.

5. **Kernel Density Estimation (KDE)**: Apply KDE to approximate the probability density functions (PDFs) of the return distribution.

6. **Visualize the Data**: Plot the data, KDE fits, histograms, and Gaussian fits of log-returns of Bitcoin in the selected time frames.

7. **Calculate Tail Areas**: Identify the tails of each distribution and calculate the area under the curves for the first and third quartiles.

8. **Compare Results**: Conduct a comparative analysis between the features of the newly defined feature, "Tails", and other statistical features such as Mean, Standard Deviation, Skewness, and Kurtosis.

9. **Analyze Trends**: Observe and analyze trends of changes in these statistical features by moving to broader time frames.

## Software and Libraries

This project uses the following Python packages: 
- `yfinance`
- `numpy`
- `matplotlib`
- `seaborn`
- `sklearn`

## Author

This project was created by Amir Gholizad, a student at Memorial University, for the course CMSC 6950 under the guidance of Dr. Scott MacLachlan, Department of Mathematics and Statistics. The project was completed in November 2023.