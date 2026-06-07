import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from osgeo import gdal
from tkinter import filedialog
from f_functions import *

def main():
    # Configuration
    sys.path.append('C:/Users/felip/Desktop/Msc-UTFSM/00_Codigos/Python')
    name = 'Laguna-Seca'
    path = f'C:/Users/felip/Desktop/Msc-UTFSM/{name}'

    # Read Data
    df_roi = read_csv_file(f'{path}/02-Space-Facilities/00-ROI-utm.csv','#')
    df_roi_list = read_csv_file(f'{path}/02-Space-Facilities/03-ROI-LIST.csv','#')
    Mcal = read_csv_file(f'{path}/Mcal_py.csv','#')

    # Extract ROI points
    if df_roi is not None:
        ip1 = df_roi[df_roi['pl'] == 'p1'].index[0]
        ip2 = df_roi[df_roi['pl'] == 'p2'].index[0]
        p1 = df_roi.loc[ip1, ['xUTM', 'yUTM']].values
        p2 = df_roi.loc[ip2, ['xUTM', 'yUTM']].values

    # Load images
    images_by_date = {}
    directory = f'{path}/02-Space-Facilities/ROI-LIST/'

    if df_roi_list is not None:
        for _, row in df_roi_list.iterrows():
            file_name = row['Ruta']
            date = row['Fecha']
            if file_name.endswith('.tif'):
                image_array = load_image(file_name)
                if image_array is not None:
                    images_by_date[date] = image_array

    # Process and visualize images
    fecha = df_roi_list['Fecha'].unique()[2]
    if fecha in images_by_date:
        ima = normalize_image(images_by_date[fecha][:, :, [3, 2, 1]])
        if ima is not None:
            fig = plot_image_with_roi_selection(ima, 'Laguna Seca')

            nameg = ['agua profunda', 'agua superficial', 'terreno natural', 'relave seco',
                     'relave consolidado', 'relave húmedo', 'otros']
            s_lam = np.arange(0, 12)
            serie = np.zeros((1, 15))

            # Interactive ROI selection (simulated since Plotly does not directly support ginput)
            # Replace with your actual ROI selection logic
            for i_Ng in range(1, len(nameg) + 1):
                print(f'{i_Ng} de {len(nameg)}: {nameg[i_Ng - 1]}')
                # Simulated ROI selection, replace with actual Plotly interaction
                # For example, add scatter points and use click events for selection
                fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers', marker=dict(color='red', size=10), name=nameg[i_Ng - 1]))

            fig.show()

if __name__ == "__main__":
    main()
