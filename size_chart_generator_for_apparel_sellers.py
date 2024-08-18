import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Set Page Config
st.set_page_config(page_title="AI-Powered Size Chart Generator", layout="wide")

# Title
st.title("🛍️ AI-Powered Size Chart Generator for Apparel Sellers")

# Load the synthetic dataset
@st.cache_data
def load_data():
    np.random.seed(42)
    n_samples = 1000
    data = pd.DataFrame({
        'Height (cm)': np.random.uniform(150, 190, n_samples),
        'Weight (kg)': np.random.uniform(50, 100, n_samples),
        'Chest (cm)': np.random.uniform(80, 120, n_samples),
        'Waist (cm)': np.random.uniform(60, 100, n_samples),
        'Hip (cm)': np.random.uniform(80, 120, n_samples),
        'Age': np.random.randint(18, 70, n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Size Purchased': np.random.choice(['XS', 'S', 'M', 'L', 'XL', 'XXL'], n_samples),
        'Return': np.random.choice([True, False], n_samples, p=[0.2, 0.8])
    })
    return data

data = load_data()

# Sidebar for User Inputs
st.sidebar.header("🔍 User Input Parameters")

def user_input_features():
    height = st.sidebar.slider('Height (cm)', 150, 190, 170)
    weight = st.sidebar.slider('Weight (kg)', 50, 100, 70)
    chest = st.sidebar.slider('Chest (cm)', 80, 120, 90)
    waist = st.sidebar.slider('Waist (cm)', 60, 100, 75)
    hip = st.sidebar.slider('Hip (cm)', 80, 120, 95)
    age = st.sidebar.slider('Age', 18, 70, 30)
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    return pd.DataFrame({
        'Height (cm)': [height],
        'Weight (kg)': [weight],
        'Chest (cm)': [chest],
        'Waist (cm)': [waist],
        'Hip (cm)': [hip],
        'Age': [age],
        'Gender': [gender]
    })

input_df = user_input_features()

# Define Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Problem Statement", "📚 Concept & Methodology", "📊 User Data Overview", "🛠️ Cluster Analysis", "📏 Size Recommendations"])

# Tab 1: Problem Statement
with tab1:
    st.header("Problem Statement")
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px;">
        <h3>AI-Powered Size Chart Generator for Apparel Sellers</h3>
        <p><strong>Problem Statement:</strong> Develop an AI system that generates accurate size charts for apparel sellers with limited or inaccurate size data, based on user body measurements and previous purchase history.</p>
        <h4>Detailed Description</h4>
        <p>Create a model that:</p>
        <ul>
            <li>Utilizes a database of user body measurements (height, weight, chest, waist, hip, etc.)</li>
            <li>Analyzes users' previous purchase history and return/exchange data</li>
            <li>Clusters similar body types and their corresponding successful purchases</li>
            <li>Generates a comprehensive size chart for sellers, including measurements for different sizes (S, M, L, XL, etc.)</li>
            <li>Provides confidence scores for each measurement in the generated size chart</li>
            <li>Allows for easy updating as new purchase data becomes available</li>
        </ul>
        <h4>Judging Criteria</h4>
        <ol>
            <li>Accuracy of generated size charts compared to brands with known accurate data</li>
            <li>Handling of different apparel categories (e.g., tops, bottoms, dresses)</li>
            <li>Effectiveness in reducing size-related returns (simulated using holdout data)</li>
            <li>Scalability and adaptability to new brands or product lines</li>
            <li>Processing speed and efficiency in generating and updating size charts</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# Tab 2: Concept & Methodology
with tab2:
    st.header("Concept & Methodology")
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px;">
        <h3>How the Objective is Achieved</h3>
        <ol>
            <li><strong>Data Collection:</strong> Simulated user body measurements and purchase history are used.</li>
            <li><strong>Similar User Identification:</strong> Nearest Neighbors algorithm finds users with similar body measurements.</li>
            <li><strong>Clustering Analysis:</strong> Machine learning algorithms group users with similar body types.</li>
            <li><strong>Size Recommendation:</strong> Based on similar users' purchase history, sizes are recommended with confidence scores.</li>
            <li><strong>Visualization:</strong> Interactive charts help users understand data distributions and recommendations.</li>
        </ol>
        
        <h3>Key Concepts</h3>
        <ul>
            <li><strong>Nearest Neighbors:</strong> Finds the most similar data points in a dataset.</li>
            <li><strong>Clustering Algorithms:</strong> Group similar data points together.</li>
            <li><strong>Confidence Scores:</strong> Indicate the reliability of size recommendations.</li>
            <li><strong>Data Normalization:</strong> Ensures all measurements are on the same scale for comparison.</li>
        </ul>
        
        <h3>Working Methodology</h3>
        <ol>
            <li>User inputs their body measurements.</li>
            <li>The system finds similar users in the database.</li>
            <li>Clustering analysis is performed to understand body type distributions.</li>
            <li>Size recommendations are generated based on similar users' successful purchases.</li>
            <li>Confidence scores are calculated to indicate the reliability of each recommendation.</li>
            <li>Results are presented visually for easy interpretation.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# Tab 3: User Data Overview
with tab3:
    st.header("User Data Overview")
    
    st.subheader("Dataset Preview")
    st.write(data.head())
    
    st.subheader('User Input Parameters')
    st.write(input_df)

    def find_similar_users(input_data, dataset, n_neighbors=5):
        features = ['Height (cm)', 'Weight (kg)', 'Chest (cm)', 'Waist (cm)', 'Hip (cm)']
        scaler = StandardScaler()
        dataset_scaled = scaler.fit_transform(dataset[features])
        input_scaled = scaler.transform(input_data[features])
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean').fit(dataset_scaled)
        distances, indices = nbrs.kneighbors(input_scaled)
        similar_users = dataset.iloc[indices[0]]
        similar_users['Similarity Score'] = 1 / (1 + distances[0])
        return similar_users

    similar_users = find_similar_users(input_df, data)

    st.subheader('Similar Users')
    st.write(similar_users)

# Tab 4: Cluster Analysis
with tab4:
    st.header('Cluster Analysis')
    
    # Clustering Algorithm Selection
    clustering_algorithm = st.selectbox(
        "Select Clustering Algorithm",
        ("KMeans", "DBSCAN", "Agglomerative")
    )

    # Clustering Parameters
    if clustering_algorithm == "KMeans":
        n_clusters = st.slider("Number of Clusters", 2, 10, 5)
    elif clustering_algorithm == "DBSCAN":
        eps = st.slider("Epsilon", 0.1, 2.0, 0.5)
        min_samples = st.slider("Min Samples", 2, 10, 5)
    else:  # Agglomerative
        n_clusters = st.slider("Number of Clusters", 2, 10, 5)
        linkage = st.selectbox("Linkage", ("ward", "complete", "average"))
    
    def cluster_data(data, algorithm, **params):
        features = ['Height (cm)', 'Weight (kg)', 'Chest (cm)', 'Waist (cm)', 'Hip (cm)']
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(data[features])
        
        if algorithm == "KMeans":
            model = KMeans(n_clusters=params['n_clusters'], random_state=42)
        elif algorithm == "DBSCAN":
            model = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
        else:  # Agglomerative
            model = AgglomerativeClustering(n_clusters=params['n_clusters'], linkage=params['linkage'])
        
        data['Cluster'] = model.fit_predict(scaled_features)
        return data

    if clustering_algorithm == "KMeans":
        clustered_data = cluster_data(data, "KMeans", n_clusters=n_clusters)
    elif clustering_algorithm == "DBSCAN":
        clustered_data = cluster_data(data, "DBSCAN", eps=eps, min_samples=min_samples)
    else:  # Agglomerative
        clustered_data = cluster_data(data, "Agglomerative", n_clusters=n_clusters, linkage=linkage)

    st.subheader('Cluster Distribution')
    cluster_distribution = clustered_data['Cluster'].value_counts().reset_index()
    cluster_distribution.columns = ['Cluster', 'Count']
    st.write(cluster_distribution)

    st.subheader('Cluster Visualization')
    fig = px.scatter(clustered_data, x='Height (cm)', y='Weight (kg)', color='Cluster',
                     hover_data=['Chest (cm)', 'Waist (cm)', 'Hip (cm)'],
                     title=f"Clusters of Users Based on Body Measurements ({clustering_algorithm})")
    st.plotly_chart(fig)

# Tab 5: Size Recommendations
with tab5:
    st.header('Size Recommendations')
    
    def generate_size_recommendation(input_data, clustered_data):
        similar_users = find_similar_users(input_data, clustered_data)
        
        if similar_users.empty:
            st.warning("No similar users found. Unable to provide size recommendations.")
            return pd.DataFrame()
        
        size_recommendations = similar_users['Size Purchased'].value_counts().reset_index()
        size_recommendations.columns = ['Recommended Size', 'Count']
        size_recommendations['Confidence Score'] = size_recommendations['Count'] / size_recommendations['Count'].sum()
        
        return size_recommendations.sort_values('Confidence Score', ascending=False)

    size_recommendations = generate_size_recommendation(input_df, clustered_data)
    
    if not size_recommendations.empty:
        st.write(size_recommendations)
        
        fig = px.bar(size_recommendations, x='Recommended Size', y='Confidence Score', 
                     title='Size Recommendations with Confidence Scores',
                     labels={'Recommended Size': 'Recommended Size', 'Confidence Score': 'Confidence Score'})
        st.plotly_chart(fig)
    else:
        st.write("No size recommendations available based on the current input.")

# Style the app with aesthetic colors
st.markdown("""
    <style>
        .stApp {
            background-color: #F0F2F6;
            font-family: 'Arial', sans-serif;
        }
        .stTabs [data-baseweb="tab-list"] button {
            color: #4B4B4B;
            background-color: #F7F9FC;
            border-bottom: 2px solid #4B4B4B;
        }
        .stTabs [data-baseweb="tab-list"] button [data-testid="stTabs-tab-active"] {
            color: #FFFFFF;
            background-color: #4B4B4B;
            border-bottom: 2px solid #4B4B4B;
        }
        .stMarkdown h2, .stMarkdown h3 {
            color: #333333;
        }
    </style>
""", unsafe_allow_html=True)