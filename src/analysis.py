import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import umap
import os

# Create output directory
output_dir = "../report/figures"
os.makedirs(output_dir, exist_ok=True)

# Load data
df = pd.read_csv("../data/processed_dataset.csv")

# Map Sleep Disorder
sleep_disorder_map = {0: 'None', 1: 'Insomnia', 2: 'Sleep Apnea'}
df['Sleep Disorder Label'] = df['Sleep Disorder'].map(sleep_disorder_map)

# 1. Class Balance
print("Generating Class Balance plot...")
class_counts = df['Sleep Disorder Label'].value_counts().reset_index()
class_counts.columns = ['Sleep Disorder', 'Count']
fig_balance = px.bar(class_counts, x='Sleep Disorder', y='Count', 
                     title='Class Balance: Sleep Disorder',
                     color='Sleep Disorder',
                     text='Count')
fig_balance.write_html(f"{output_dir}/class_balance.html")
fig_balance.write_image(f"{output_dir}/class_balance.png", scale=3)

# 2. Correlation Matrix
print("Generating Correlation Matrix...")
# Drop the label column we just created for correlation
corr_matrix = df.drop('Sleep Disorder Label', axis=1).corr()
fig_corr = px.imshow(corr_matrix, 
                     text_auto=True, 
                     aspect="auto",
                     title='Correlation Matrix',
                     color_continuous_scale='RdBu_r')
fig_corr.write_html(f"{output_dir}/correlation_matrix.html")
fig_corr.write_image(f"{output_dir}/correlation_matrix.png", scale=3)

# 3. Feature Importance
print("Calculating Feature Importance...")
X = df.drop(['Sleep Disorder', 'Sleep Disorder Label'], axis=1)
y = df['Sleep Disorder']

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

fig_imp = px.bar(importances, x='Importance', y='Feature', orientation='h',
                 title='Feature Importance (Random Forest)',
                 color='Importance',
                 color_continuous_scale='Viridis')
fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
fig_imp.write_html(f"{output_dir}/feature_importance.html")
fig_imp.write_image(f"{output_dir}/feature_importance.png", scale=3)

# 4. Distribution of Age by Sleep Disorder (Using Raw Data)
print("Generating Age Distribution plot...")
# Load raw data to get actual Age values (not normalized)
df_raw = pd.read_csv("../data/raw_dataset.csv")
df_raw['Sleep Disorder'] = df_raw['Sleep Disorder'].fillna('None')

fig_age = px.box(df_raw, x="Sleep Disorder", y="Age", 
                 color="Sleep Disorder",
                 title="Distribution of Age by Sleep Disorder",
                 points="all") # Show all points
fig_age.write_html(f"{output_dir}/age_distribution.html")
fig_age.write_image(f"{output_dir}/age_distribution.png", scale=3)

fig_age.write_image(f"{output_dir}/age_distribution.png", scale=3)

# 5. PCA 3D Visualization
print("Generating PCA 3D plot...")
# Use normalized data (df) but drop labels AND Occupation columns
cols_to_drop = ['Sleep Disorder', 'Sleep Disorder Label'] + [col for col in df.columns if col.startswith('Occupation_')]
X_pca = df.drop(cols_to_drop, axis=1)

pca = PCA(n_components=3)
components = pca.fit_transform(X_pca)

fig_pca = px.scatter_3d(
    components, x=0, y=1, z=2,
    color=df['Sleep Disorder Label'],
    title='PCA 3D Visualization',
    labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'},
    opacity=0.7
)
fig_pca.write_html(f"{output_dir}/pca_3d.html")
fig_pca.write_image(f"{output_dir}/pca_3d.png", scale=3)

# 6. UMAP 3D Visualization
print("Generating UMAP 3D plot...")
reducer = umap.UMAP(n_components=3, random_state=42)
embedding = reducer.fit_transform(X_pca)

fig_umap = px.scatter_3d(
    embedding, x=0, y=1, z=2,
    color=df['Sleep Disorder Label'],
    title='UMAP 3D Visualization',
    labels={'0': 'UMAP 1', '1': 'UMAP 2', '2': 'UMAP 3'},
    opacity=0.7
)
fig_umap.write_html(f"{output_dir}/umap_3d.html")
fig_umap.write_image(f"{output_dir}/umap_3d.png", scale=3)
