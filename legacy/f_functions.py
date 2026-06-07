## Este archivo contiene las siguientes funciones:
#load_image(file_path)
#f_rgb2hsl(rgb_in)
#read_csv_file(file_path, comment='#')
#normalize_image(image_array) a colores RGB 255
#plot_image_with_roi_selection(image, title)


# Función para cargar una imagen y convertirla en un np.array

def load_image(file_path):
    from osgeo import gdal
    import numpy as np
    try:
        dataset = gdal.Open(file_path, gdal.GA_ReadOnly)
        if not dataset:
            raise FileNotFoundError(f'No se pudo abrir el archivo: {file_path}')
        num_bands = dataset.RasterCount
        x_size = dataset.RasterXSize
        y_size = dataset.RasterYSize
        image_array = np.zeros((y_size, x_size, num_bands), dtype=np.float32)
        for i in range(num_bands):
            band = dataset.GetRasterBand(i + 1)
            image_array[:, :, i] = band.ReadAsArray()
        return image_array
    except Exception as e:
        print(f'Error al cargar la imagen: {e}')
        return None

# Función para cargar una archivo csv y convertirlo a dataframe    
def read_csv_file(file_path, comment='#'):
    import pandas as pd
    try:
        return pd.read_csv(file_path, comment=comment)
    except Exception as e:
        print(f'Error al leer el archivo CSV: {e}')
        return None


def f_rgb2hsl(rgb_in):
    import numpy as np
    import matplotlib
    """
    Convierte un color RGB (Red, Green, Blue) en un color HSL (Hue, Saturation, Lightness).
    Entrada:
        rgb (array_like): Un arreglo de numpy o una lista que representa el color RGB.
    Salida:
        hsl (array_like): Un arreglo de numpy que representa el color HSL.
    """
    # Normalizar los valores RGB
    rgb = np.reshape(rgb_in, (-1, 3))
    #r, g, b = np.array(rgb) / 255.0
    # Calcula el máximo y mínimo de los tres colores RGB
    mx = np.max(rgb, axis=1)
    mn = np.min(rgb, axis=1)
    # Calcula la luminosidad
    L = (mx + mn) / 2.0
    # Inicializa la saturación
    S = np.zeros_like(L)
    # Calcula la saturación para los casos donde L <= 0.5
    lowlidx = L <= 0.5
    calc = np.zeros_like(L)
    calc[lowlidx] = (mx[lowlidx] - mn[lowlidx]) / (mx[lowlidx] + mn[lowlidx])
    # Calcula la saturación para los casos donde L > 0.5
    hilidx = L > 0.5
    calc[hilidx] = (mx[hilidx] - mn[hilidx]) / (2 - (mx[hilidx] + mn[hilidx]))
    # Asigna los valores calculados de saturación
    S[lowlidx | hilidx] = calc[lowlidx | hilidx]
    # Utiliza la función rgb2hsv para obtener el matiz
    hsv = matplotlib.colors.rgb_to_hsv(rgb)
    H = hsv[:, 0]
    # Concatena los componentes HSL
    hsl = np.column_stack((H, S, L))
    # Redondea los valores y reordena la matriz a la forma original
    hsl = np.round(hsl * 100000) / 100000
    hsl = np.reshape(hsl, np.shape(rgb_in))
    return hsl

def f_hsl2rgb(hsl_in):
    import numpy as np

    """
    Convierte un color HSL (Hue, Saturation, Lightness) en un color RGB (Red, Green, Blue).
    Entrada:
        hsl (array_like): Un arreglo de numpy o una lista que representa el color HSL.
                          H está en el rango [0, 360], S y L en el rango [0, 100].
    Salida:
        rgb (array_like): Un arreglo de numpy que representa el color RGB.
    """
    # Normalizar los valores HSL a [0, 1]
    hsl = np.reshape(hsl_in, (-1, 3))
    H = hsl[:, 0] / 360.0  # H entre 0 y 1
    S = hsl[:, 1] / 100.0  # S entre 0 y 1
    L = hsl[:, 2] / 100.0  # L entre 0 y 1
    
    # Inicializa R, G, B
    R = np.zeros_like(H)
    G = np.zeros_like(H)
    B = np.zeros_like(H)
    
    # Casos donde la saturación es cero
    nullSidx = S == 0
    R[nullSidx] = L[nullSidx]
    G[nullSidx] = L[nullSidx]
    B[nullSidx] = L[nullSidx]
    
    # Casos donde la saturación no es cero
    nonnullSidx = S != 0
    if np.any(nonnullSidx):
        def _hue2rgb(p, q, t):
            t = t % 1
            t = np.where(t < 0, t + 1, t)
            t = np.where(t > 1, t - 1, t)

            return (
                np.where(t < 1/6, p + (q - p) * 6 * t,
                np.where(t < 1/2, q,
                np.where(t < 2/3, p + (q - p) * (2/3 - t) * 6, p)))
            )
        
        q = np.where(L < 0.5, L * (1 + S), L + S - L * S)
        p = 2 * L - q
        
        R[nonnullSidx] = _hue2rgb(p[nonnullSidx], q[nonnullSidx], H[nonnullSidx] + 1/3)
        G[nonnullSidx] = _hue2rgb(p[nonnullSidx], q[nonnullSidx], H[nonnullSidx])
        B[nonnullSidx] = _hue2rgb(p[nonnullSidx], q[nonnullSidx], H[nonnullSidx] - 1/3)
    
    # Concatena los componentes RGB
    rgb = np.column_stack((R, G, B))
    
    # Redondea los valores y reordena la matriz a la forma original
    rgb = np.round(rgb * 100000) / 100000
    rgb = np.reshape(rgb, np.shape(hsl_in))
    
    # Escalar RGB a rango [0, 255]
    #rgb = np.clip(rgb * 255, 0, 255).astype(int)
    #RETURN RGB ENTRE 0 Y 1
    return rgb

def normalize_image(image_array):
    import numpy as np
    try:
        for i in range(image_array.shape[2]):
            image_array[:, :, i] = np.floor(255. * image_array[:, :, i] / np.max(image_array[:, :, i]))
        return image_array.astype(np.uint8)
    except Exception as e:
        print(f'Error al normalizar la imagen: {e}')
        return None

def plot_image_with_roi_selection(image, title):
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Image(z=image))
    fig.update_layout(
        title=title,
        xaxis_title='X [m]',
        yaxis_title='Y [m]',
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(constrain="domain")
    )
    return fig


def plot_pixel_vs_clases(pixel_index, refle_data, MR_ref_mod, nameg=None):
    import matplotlib.pyplot as plt
    from sklearn.metrics.pairwise import cosine_similarity
    """
    Visualiza la distancia de un solo píxel respecto a todas las clases en el plano
    (1 - cos(θ), distancia euclidiana), e indica el radio combinado.
    
    Parámetros:
    - pixel_index: índice entero del píxel en refle_data.
    - refle_data: matriz [n_pix, N] con reflectancias por píxel.
    - MR_ref_mod: matriz [Ng, N] con firmas espectrales de referencia.
    - nameg: lista con nombres de clases (opcional).
    """
    Ng = MR_ref_mod.shape[0]
    MR_ref_mod_norm = MR_ref_mod / np.linalg.norm(MR_ref_mod, axis=1, keepdims=True)
    
    v = refle_data[pixel_index]
    v_norm = v / np.linalg.norm(v)
    v_norm = v_norm.reshape(1, -1)

    cos_sim = cosine_similarity(v_norm, MR_ref_mod_norm).flatten()
    
    cos_dissim = 1 - cos_sim
    dists = np.linalg.norm(MR_ref_mod/10000 - v, axis=1)
    radios = np.sqrt(cos_dissim**2 + dists**2)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(Ng):
        label = nameg[i] if nameg else f"Clase {i}"
        ax.scatter(cos_dissim[i], dists[i], s=100, label=f"{label} (R={radios[i]:.3f})")

    ax.set_xlabel("1 - cos(θ) (Disimilaridad Angular)")
    ax.set_ylabel("Distancia Euclidiana")
    ax.set_title(f"Comparación de espectro del píxel #{pixel_index} con clases")
    ax.grid(True)
    ax.legend(loc="best", fontsize="small")
    plt.tight_layout()
    plt.show()
