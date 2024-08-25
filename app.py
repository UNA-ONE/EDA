import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.impute import SimpleImputer
import numpy as np
from scipy import stats

st.set_page_config(page_title='EDA Portal', page_icon='📊')

# Welcome Text and Instructions
st.title('Welcome to the EDA Portal! 🌟')
st.write("""
    Explore our powerful EDA (Exploratory Data Analysis) tool to gain insights from your data. 
    Here’s what you can do:
    
    1. **Upload Your Data**: Start by uploading a CSV or Excel file to begin analyzing your data.
    2. **View and Summarize**: Get a summary of your dataset, including basic statistics, data types, and a list of columns.
    3. **Handle Missing Values**: Identify and handle missing values with various imputation methods.
    4. **Detect Outliers**: Use Z-Score or IQR methods to detect and handle outliers in your data.
    5. **Visualize Data**: Create various charts and graphs to visualize your data and reveal trends.
    6. **Group and Aggregate**: Group data by columns and perform aggregation operations for deeper insights.
    7. **Download Updated Data**: After making changes, download the updated dataset for further use.

    Feel free to upload your own dataset or explore our default dataset to get a feel for our tool!
""")

st.title(':rainbow[Your GoTo EDA Portal]')
st.subheader(':gray[Explore Data with ease.]', divider='rainbow')

# Define a default dataset (example data)
DEFAULT_DATA = pd.DataFrame({
    'Category': ['A', 'B', 'A', 'C', 'B', 'A', 'C'],
    'Value': [10, 20, 10, 30, 20, 10, 30]
})

# Function to load data
def load_data(file):
    if file is not None:
        if file.name.endswith('csv'):
            return pd.read_csv(file)
        else:
            return pd.read_excel(file)
    return None

# Create a placeholder for the uploaded data
file = st.file_uploader('Drop csv or excel file', type=['csv', 'xlsx'])
data = load_data(file)

# Use default data if no file is uploaded
if data is None:
    data = DEFAULT_DATA
    st.info('Showing default data. Upload your own data to replace this.')

st.dataframe(data)

# Rest of your code...

st.subheader(':rainbow[Basic information of the dataset]', divider='rainbow')
tab1, tab2, tab3, tab4, tab5 = st.tabs(['Summary', 'Top and Bottom Rows', 'Data Types', 'Columns', 'Descriptive Statistics'])

with tab1:
    st.write(f'There are {data.shape[0]} rows in dataset and  {data.shape[1]} columns in the dataset')
    st.subheader(':gray[Statistical summary of the dataset]')
    st.dataframe(data.describe())
with tab2:
    st.subheader(':gray[Top Rows]')
    toprows = st.slider('Number of rows you want', 1, data.shape[0], key='topslider')
    st.dataframe(data.head(toprows))
    st.subheader(':gray[Bottom Rows]')
    bottomrows = st.slider('Number of rows you want', 1, data.shape[0], key='bottomslider')
    st.dataframe(data.tail(bottomrows))
with tab3:
    st.subheader(':grey[Data types of column]')
    st.dataframe(data.dtypes)
with tab4:
    st.subheader('Column Names in Dataset')
    st.write(list(data.columns))
with tab5:
    st.subheader(':gray[Descriptive Statistics]')
    
    # Function to calculate descriptive statistics
    def descriptive_statistics(df):
        stats_df = pd.DataFrame()
        stats_df['mean'] = df.mean()
        stats_df['std_dev'] = df.std()
        stats_df['min'] = df.min()
        stats_df['25%'] = df.quantile(0.25)
        stats_df['median'] = df.median()
        stats_df['75%'] = df.quantile(0.75)
        stats_df['max'] = df.max()
        stats_df['variance'] = df.var()
        stats_df['skewness'] = df.skew()
        stats_df['kurtosis'] = df.kurtosis()
        return stats_df

    # Calculate descriptive statistics for numeric columns
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    if numeric_columns.empty:
        st.write("No numeric columns available for descriptive statistics.")
    else:
        stats_df = descriptive_statistics(data[numeric_columns])
        st.dataframe(stats_df)

   

    st.subheader(':rainbow[Handle Missing Values]', divider='rainbow')

    # Create a temporary DataFrame to store the modified data
    if 'temp_data' not in st.session_state:
        st.session_state.temp_data = data.copy()

    # Calculate missing values
    def calculate_missing_values(df):
        missing_data = df.isnull().sum().reset_index()
        missing_data.columns = ['Column', 'Missing Values']
        return missing_data[missing_data['Missing Values'] > 0]

    missing_data = calculate_missing_values(st.session_state.temp_data)

    if missing_data.empty:
        st.write("No missing values found in the dataset.")
    else:
        st.write("Missing values detected in the following columns:")
        st.dataframe(missing_data)

        recommendations = []
        for column in missing_data['Column']:
            if st.session_state.temp_data[column].dtype in ['float64', 'int64']:
                recommendations.append(f"💡 The column '{column}' contains numerical data. Imputing with the mean might be the best option.")
            elif st.session_state.temp_data[column].dtype == 'object':
                recommendations.append(f"💡 The column '{column}' contains categorical data. Imputing with the most frequent value might be the best option.")
        
        for recommendation in recommendations:
            st.write(recommendation)

        with st.expander('Missing Value Imputation'):
            col1, col2, col3 = st.columns(3)

            with col1:
                column = st.selectbox('Select Column', options=missing_data['Column'], key='impute_column', help='Select the column to impute missing values for.')

            with col2:
                method = st.selectbox('Select Imputation Method',
                                      options=['Mean', 'Median', 'Most Frequent', 'Constant'],
                                      key='impute_method',
                                      help='Choose the method to impute missing values.')

            with col3:
                if method == 'Constant':
                    fill_value = st.text_input('Value for Constant Imputation', key='constant_value', help='Specify the constant value for imputation.')
                else:
                    fill_value = None

            impute = st.button('Impute', key='impute_button')

            if impute:
                imputer = None
                if method == 'Mean':
                    imputer = SimpleImputer(strategy='mean')
                elif method == 'Median':
                    imputer = SimpleImputer(strategy='median')
                elif method == 'Most Frequent':
                    imputer = SimpleImputer(strategy='most_frequent')
                elif method == 'Constant' and fill_value is not None:
                    imputer = SimpleImputer(strategy='constant', fill_value=fill_value)

                if imputer:
                    st.session_state.temp_data[[column]] = imputer.fit_transform(st.session_state.temp_data[[column]])
                    st.success(f'Missing values in column "{column}" have been imputed using the {method} method.')
                    
                    # Show top 7 rows where data was imputed
                    st.subheader("Top 7 Rows After Imputation")
                    imputed_data = st.session_state.temp_data.head(7)
                    st.dataframe(imputed_data)

                    # Recalculate missing values after imputation
                    missing_data = calculate_missing_values(st.session_state.temp_data)
                    
                    if missing_data.empty:
                        st.write("All missing values have been imputed.")
                    else:
                        st.write("Remaining missing values:")
                        st.dataframe(missing_data)

                    # Offer to download the updated data if all columns have been handled
                    if missing_data.empty:
                        csv = st.session_state.temp_data.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Updated Data",
                            data=csv,
                            file_name='updated_data.csv',
                            mime='text/csv'
                        )

    # Outlier Detection Section
    st.subheader(':rainbow[Outlier Detection]', divider='rainbow')

    def detect_outliers_zscore(df, column):
        z_scores = np.abs(stats.zscore(df[column].dropna()))
        return df[z_scores > 3]

    def detect_outliers_iqr(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        return df[(df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))]

    def detect_and_display_outliers():
        numeric_columns = st.session_state.temp_data.select_dtypes(include=['float64', 'int64']).columns
        if numeric_columns.empty:
            st.write("No numeric columns available for outlier detection.")
        else:
            column = st.selectbox('Select Numeric Column for Outlier Detection', options=numeric_columns, key='outlier_detection_column')

            detection_method = st.selectbox('Select Outlier Detection Method', options=['Z-Score', 'IQR'],
                                            key='outlier_detection_method',
                                            help='Choose the method for outlier detection. [Z-Score Documentation](https://www.analyticsvidhya.com/blog/2022/08/dealing-with-outliers-using-the-z-score-method/) [IQR Documentation](https://www.analyticsvidhya.com/blog/2022/09/dealing-with-outliers-using-the-iqr-method/)')

            if st.button('Detect Outliers', key='outlier_detection_button'):
                if detection_method == 'Z-Score':
                    outliers = detect_outliers_zscore(st.session_state.temp_data, column)
                elif detection_method == 'IQR':
                    outliers = detect_outliers_iqr(st.session_state.temp_data, column)

                if not outliers.empty:
                    st.subheader(f"Outliers Detected in Column '{column}'")
                    st.dataframe(outliers)

                    # Visualization
                    fig = px.box(st.session_state.temp_data, y=column, title=f'Box Plot of {column}')
                    st.plotly_chart(fig)

                    # Recommendations
                    st.write("💡 Recommendations for Handling Outliers:")
                    st.write("1. **Remove Outliers**: If outliers are due to data entry errors, consider removing them.")
                    st.write("2. **Transform Data**: Apply transformations such as logarithmic scaling to reduce the impact of outliers.")
                    st.write("3. **Cap or Floor Values**: Set upper and lower bounds to cap or floor extreme values.")

                    # Note and Documentation Links
                    st.write("🔧 We are working on bringing you a dynamic outlier remover. Stay tuned!")
                    st.write("[Learn more about Z-Score](https://www.analyticsvidhya.com/blog/2022/08/dealing-with-outliers-using-the-z-score-method/) and [IQR](https://www.analyticsvidhya.com/blog/2022/09/dealing-with-outliers-using-the-iqr-method/) methods")

                else:
                    st.write("No outliers detected.")

    detect_and_display_outliers()



st.subheader(':green[Analytics Section]', divider='green')

# Display an image
# st.image('analytics.jpg', caption='Basic Analytics Section', use_column_width=True, width=200)





st.subheader(':rainbow[Column Values To Count]', divider='rainbow')
with st.expander('Value Count'):
    col1, col2 = st.columns(2)
    with col1:
        column = st.selectbox('Choose Column name', options=list(data.columns), key='value_count_column')
    with col2:
        toprows = st.number_input('Top rows', min_value=1, step=1, key='value_count_toprows')

    count = st.button('Count', key='value_count_button')
    if count:
        result = data[column].value_counts().reset_index().head(toprows)
        result.columns = [column, 'count']  # Rename columns
        
        st.dataframe(result)
        st.subheader('Visualization', divider='gray')
        
        fig = px.bar(data_frame=result, x=column, y='count', text='count', template='plotly_white')
        st.plotly_chart(fig)
        
        fig = px.line(data_frame=result, x=column, y='count', text='count', template='plotly_white')
        st.plotly_chart(fig)
        
        fig = px.pie(data_frame=result, names=column, values='count')
        st.plotly_chart(fig)







st.subheader(':rainbow[Groupby : Simplify your data analysis]', divider='rainbow')
st.write('The groupby lets you summarize data by specific categories and groups')
with st.expander('Group By your columns'):
    col1, col2, col3 = st.columns(3)
    with col1:
        groupby_cols = st.multiselect('Choose your column to groupby', options=list(data.columns))
    with col2:
        operation_col = st.selectbox('Choose column for operation', options=list(data.columns))
    with col3:
        operation = st.selectbox('Choose operation', options=['sum', 'max', 'min', 'mean', 'median', 'count'])
    
    if groupby_cols:
        result = data.groupby(groupby_cols).agg(
            newcol=(operation_col, operation)
        ).reset_index()

        st.dataframe(result)

        st.subheader(':gray[Data Visualization]', divider='gray')
        graphs = st.selectbox('Choose your graphs', options=['line', 'bar', 'scatter', 'pie', 'sunburst'])
        if graphs == 'line':
            x_axis = st.selectbox('Choose X axis', options=list(result.columns))
            y_axis = st.selectbox('Choose Y axis', options=list(result.columns))
            color = st.selectbox('Color Information', options=[None] + list(result.columns))
            fig = px.line(data_frame=result, x=x_axis, y=y_axis, color=color, markers='o')
            st.plotly_chart(fig)
        elif graphs == 'bar':
            x_axis = st.selectbox('Choose X axis', options=list(result.columns))
            y_axis = st.selectbox('Choose Y axis', options=list(result.columns))
            color = st.selectbox('Color Information', options=[None] + list(result.columns))
            facet_col = st.selectbox('Column Information', options=[None] + list(result.columns))
            fig = px.bar(data_frame=result, x=x_axis, y=y_axis, color=color, facet_col=facet_col, barmode='group')
            st.plotly_chart(fig)
        elif graphs == 'scatter':
            x_axis = st.selectbox('Choose X axis', options=list(result.columns))
            y_axis = st.selectbox('Choose Y axis', options=list(result.columns))
            color = st.selectbox('Color Information', options=[None] + list(result.columns))
            size = st.selectbox('Size Column', options=[None] + list(result.columns))
            fig = px.scatter(data_frame=result, x=x_axis, y=y_axis, color=color, size=size)
            st.plotly_chart(fig)
        elif graphs == 'pie':
            values = st.selectbox('Choose Numerical Values', options=list(result.columns))
            names = st.selectbox('Choose labels', options=list(result.columns))
            fig = px.pie(data_frame=result, values=values, names=names)
            st.plotly_chart(fig)
        elif graphs == 'sunburst':
            path = st.multiselect('Choose your Path', options=list(result.columns))
            fig = px.sunburst(data_frame=result, path=path, values='newcol')
            st.plotly_chart(fig)









