import os
import sys
from pathlib import Path
parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)

import zipfile
import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from branca.colormap import LinearColormap
from streamlit_folium import st_folium
from src.config import DATA_DIR
from src.inference import fetch_next_hour_predictions, load_batch_of_features_from_store
from src.plot_utils import plot_prediction
import pytz

os.environ['SHAPE_RESTORE_SHX'] = 'YES'

# Initialize session state for the map
if "map_created" not in st.session_state:
    st.session_state.map_created = False

def load_zone_lookup(csv_path):
    return pd.read_csv(csv_path).set_index('LocationID')['Zone'].to_dict()

def visualize_predicted_demand(shapefile_path, predicted_demand):
    gdf = gpd.read_file(shapefile_path).to_crs("epsg:4326")
    if "LocationID" not in gdf.columns:
        raise ValueError("Shapefile must contain a 'LocationID' column to match taxi zones.")
    gdf["predicted_demand"] = gdf["LocationID"].map(predicted_demand).fillna(0)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    gdf.plot(
        column="predicted_demand",
        cmap="OrRd",
        linewidth=0.8,
        ax=ax,
        edgecolor="black",
        legend=True,
        legend_kwds={"label": "Predicted Rides", "orientation": "vertical"},
    )
    ax.set_title("Predicted NYC Taxi Rides by Zone", fontsize=16)
    ax.set_axis_off()
    st.pyplot(fig)

def create_taxi_map(shapefile_path, prediction_data):
    nyc_zones = gpd.read_file(shapefile_path)
    nyc_zones = nyc_zones.merge(
        prediction_data[["pickup_location_id", "predicted_demand"]],
        left_on="LocationID",
        right_on="pickup_location_id",
        how="left",
    )
    nyc_zones["predicted_demand"] = nyc_zones["predicted_demand"].fillna(0)
    nyc_zones = nyc_zones.to_crs(epsg=4326)
    m = folium.Map(location=[40.7128, -74.0060], zoom_start=10, tiles="cartodbpositron")
    colormap = LinearColormap(
        colors=["#FFEDA0", "#FED976", "#FEB24C", "#FD8D3C", "#FC4E2A", "#E31A1C", "#BD0026"],
        vmin=nyc_zones["predicted_demand"].min(),
        vmax=nyc_zones["predicted_demand"].max(),
    )
    colormap.add_to(m)
    def style_function(feature):
        predicted_demand = feature["properties"].get("predicted_demand", 0)
        return {
            "fillColor": colormap(float(predicted_demand)),
            "color": "black",
            "weight": 1,
            "fillOpacity": 0.7,
        }
    zones_json = nyc_zones.to_json()
    folium.GeoJson(
        zones_json,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=["zone", "predicted_demand"],
            aliases=["Zone:", "Predicted Demand:"],
            style="background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;",
        ),
    ).add_to(m)
    st.session_state.map_obj = m
    st.session_state.map_created = True
    return m

def load_shape_data_file(data_dir, url="https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip", log=True):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "taxi_zones.zip"
    extract_path = data_dir / "taxi_zones"
    shapefile_path = extract_path / "taxi_zones.shp"
    if not zip_path.exists():
        if log:
            print(f"Downloading file from {url}...")
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            with open(zip_path, "wb") as f:
                f.write(response.content)
            if log:
                print(f"File downloaded and saved to {zip_path}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to download file from {url}: {e}")
    else:
        if log:
            print(f"File already exists at {zip_path}, skipping download.")
    if not shapefile_path.exists():
        if log:
            print(f"Extracting files to {extract_path}...")
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_path)
            if log:
                print(f"Files extracted to {extract_path}")
        except zipfile.BadZipFile as e:
            raise Exception(f"Failed to extract zip file {zip_path}: {e}")
    else:
        if log:
            print(f"Shapefile already exists at {shapefile_path}, skipping extraction.")
    if log:
        print(f"Loading shapefile from {shapefile_path}...")
    try:
        gdf = gpd.read_file(shapefile_path).to_crs("epsg:4326")
        if log:
            print("Shapefile successfully loaded.")
        return gdf
    except Exception as e:
        raise Exception(f"Failed to load shapefile {shapefile_path}: {e}")

def main():
    st.title("New York Yellow Taxi Cab Demand Next Hour")
    est_timezone = pytz.timezone("America/New_York")
    current_date_est = pd.Timestamp.now(tz="Etc/UTC").tz_convert(est_timezone)
    st.header(f'{current_date_est.strftime("%Y-%m-%d %H:%M:%S")} (EST)')
    progress_bar = st.sidebar.header("Working Progress")
    progress_bar = st.sidebar.progress(0)
    N_STEPS = 5

    with st.spinner(text="Loading zone lookup"):
        zone_lookup = load_zone_lookup(DATA_DIR / "taxi_zone_lookup.csv")
        st.sidebar.write("Zone lookup loaded")
        progress_bar.progress(1 / N_STEPS)

    with st.spinner(text="Download shape file for taxi zones"):
        geo_df = load_shape_data_file(DATA_DIR)
        st.sidebar.write("Shape file was downloaded")
        progress_bar.progress(2 / N_STEPS)

    with st.spinner(text="Fetching batch of inference data"):
        features = load_batch_of_features_from_store(current_date_est)
        st.sidebar.write("Inference features fetched from the store")
        progress_bar.progress(3 / N_STEPS)

    with st.spinner(text="Fetching predictions"):
        predictions = fetch_next_hour_predictions()
        st.sidebar.write("Model was loaded from the registry")
        progress_bar.progress(4 / N_STEPS)

    shapefile_path = DATA_DIR / "taxi_zones" / "taxi_zones.shp"

    with st.spinner(text="Plot predicted rides demand"):
        st.subheader("Taxi Ride Predictions Map")
        map_obj = create_taxi_map(shapefile_path, predictions)
        if st.session_state.map_created:
            st_folium(st.session_state.map_obj, width=800, height=600, returned_objects=[])

        st.subheader("Prediction Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Rides", f"{predictions['predicted_demand'].mean():.0f}")
        with col2:
            st.metric("Maximum Rides", f"{predictions['predicted_demand'].max():.0f}")
        with col3:
            st.metric("Minimum Rides", f"{predictions['predicted_demand'].min():.0f}")

        predictions['zone_name'] = predictions['pickup_location_id'].map(zone_lookup)

        st.subheader("Top 10 Predicted Demand Locations")
        top10_predictions = predictions.sort_values("predicted_demand", ascending=False).head(10)
        st.dataframe(top10_predictions[["pickup_location_id", "zone_name", "predicted_demand"]])

        location_names = sorted(predictions['zone_name'].unique().tolist())
        selected_location_name = st.selectbox("Select a Location:", location_names)
        selected_location_id = predictions[predictions['zone_name'] == selected_location_name]['pickup_location_id'].iloc[0]

        st.subheader(f"Predicted Demand for {selected_location_name}")
        fig = plot_prediction(
            features=features[features["pickup_location_id"] == selected_location_id],
            prediction=predictions[predictions["pickup_location_id"] == selected_location_id],
        )
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)

        st.sidebar.write("Finished plotting taxi rides demand")
        progress_bar.progress(5 / N_STEPS)

if __name__ == "__main__":
    main()
