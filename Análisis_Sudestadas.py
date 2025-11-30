#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 11:27:32 2025

@author: Estudiante
"""

"""
Trabajo Final Laboratorio de Procesamiento de Información Meteorológica - DCAO UBA

=================================================
     Análisis de eventos de Sudestada 
=================================================

Este script identifica eventos de sudestada (altura del río > 2.5 m) y analiza 
su evolución utilizando datos hidrométricos y campos de presión y viento.
El objetivo es describir la estructura sinóptica típica asociada a las 
sudestadas en el Río de la Plata.

Autor: Natalia Belen Rodeiro
Fecha de Creación: 2025-11
"""

# Importamos las librerías a utilizar 

# Importamos las librerias a utilizar 
import os
import pandas as pd
import netCDF4 as nc
from netCDF4 import num2date
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# Verificamos el directorio acutal 
os.getcwd()

# Seleccionamos el nuevo directorio de trabajo
os.chdir("C:/Users/Natalia/Desktop/Entregas_Labo/TP final")

# Cargamos el archivo .xlsx - que contiene los datos de fechas y alturas (en cm) - en un DataFrame
df = pd.read_excel("C:/Users/Natalia/Desktop/Entregas_Labo/TP final/Alt_Palermo.xlsx")

# Tenemos datos en hora local Argentina, vamos a transformarlos a UTC
# Indicamos que los valores son los correspondientes a la zona horaria de Argentina
df["fecha"] = df["fecha"].dt.tz_localize("America/Argentina/Buenos_Aires")

# Convertimos a horario UTC
df["fecha"] = df["fecha"].dt.tz_convert("UTC")

# Eliminamos la información de la zona horaria para compatibilidad con Excel
df["fecha"] = df["fecha"].dt.tz_localize(None)

# Queremos seleccionar los eventos de altura de Rio mayores a 2.5m
# Agregamos una columna con la altura en metros
df["Altura (m)"] = df["alt (cm)"]/100

# Agregamos una columna con información sobre si corresponden a eventos de altura mayores a 2.5m
df["Sudestada"] = df["Altura (m)"] > 2.5

# Resumimos la información de cada evento en una tabla (fechas, duración, altura máxima alcanzada)
# Tenemos datos inviduales, tenemos que agrupar los que corresponden al mismo evento para determinar su duración 

eventos = []             # Creamos una lista vacía donde guardar los eventos
en_evento = False        # Indicador de evento
inicio = None            # Guarda la fecha
max_altura = 0           # Guarda la altura máxima alcanzada

for i in range(len(df)):

    fila = df.iloc[i]              # Acceso por índice
    es_sudestada = fila["Sudestada"]     
    altura = fila["Altura (m)"]    
    
    if es_sudestada and not en_evento: # Sudestada y NO estábamos en evento
        en_evento = True
        inicio = fila["fecha"]     # Marca el inicio
        max_altura = altura        # Primer valor del evento
    
    elif es_sudestada and en_evento: # Seguimos en el evento
        if altura > max_altura:
            max_altura = altura
    
    elif (not es_sudestada) and en_evento: # Termina un evento 
        fin = df.iloc[i-1]["fecha"]     # Última fila True
        duracion = fin - inicio
        eventos.append([inicio, fin, duracion, max_altura])
        
        en_evento = False               # Cerramos evento

# Si terminó el archivo estando en evento
if en_evento:
    fin = df.iloc[-1]["fecha"]
    duracion = fin - inicio
    eventos.append([inicio, fin, duracion, max_altura])

# Convertimos lista a DataFrame
tabla_eventos = pd.DataFrame(eventos,
                             columns=["Inicio", "Fin", "Duración", "Altura máxima alcanzada"])

# Mostramos la tabla en la consola
print(tabla_eventos)

# Agregamos una columna que muestre la duración en horas (no en días como la obtenida por la definición de duración)
tabla_eventos["Duración (horas)"] = tabla_eventos["Duración"].dt.total_seconds() / 3600

# Reordenar columnas:
tabla_eventos = tabla_eventos[
    ["Inicio", "Fin", "Duración (horas)", "Altura máxima alcanzada"]
]

# Guardamos los datos en un archivo 
tabla_eventos.to_excel("Eventos_Sudestadas_Palermo.xlsx", index=False)

# A partir de datos de presión y viento en superficie, seleccionaremos las fechas de ocurrencia de los eventos y tomaremos el punto más cercano a Buenos Aires. Graficaremos la serie temporal de presión e intensidad de viento desde 2 días antes de la ocurrencia del evento hasta 2 días después, para cada uno de los eventos encontrados.

# Cargamos los archivos nc que contienen los datos de viento y presión
data_u_viento = nc.Dataset("C:/Users/Natalia/Desktop/Entregas_Labo/TP final/uwind_2009_2012.nc")
data_v_viento = nc.Dataset("C:/Users/Natalia/Desktop/Entregas_Labo/TP final/vwind_2009_2012.nc")
data_presion = nc.Dataset("C:/Users/Natalia/Desktop/Entregas_Labo/TP final/pres_2009_2012.nc")

# Visualizamos la metadata asociada a cada archivo
print(data_u_viento)
print(data_v_viento)
print(data_presion)

# Visualizamos la metadata de las dimensiones
print(data_u_viento.dimensions)
print(data_v_viento.dimensions)
print(data_presion.dimensions)

# Visualizamos la metadata de las variables 
print(data_u_viento.variables)
print(data_v_viento.variables)
print(data_presion.variables)

# Visualizamos las claves para acceder a las variables
print(data_u_viento.variables.keys()) # dict_keys(['lat', 'lon', 'time', 'uwnd'])
print(data_v_viento.variables.keys()) # dict_keys(['lat', 'lon', 'time', 'vwnd'])
print(data_presion.variables.keys()) # dict_keys(['lat', 'lon', 'time', 'pres'])
 
# Accedemos a los datos de una variable en particular
u_viento = data_u_viento.variables["uwnd"]
v_viento = data_v_viento.variables["vwnd"]
presion = data_presion.variables["pres"]
lats = data_u_viento.variables["lat"]
lons = data_u_viento.variables["lon"]
time = data_u_viento.variables["time"]

print(u_viento.units, u_viento.dimensions, u_viento.name, u_viento.long_name, sep=" , ")

# Extraemos los datos 
u_viento = data_u_viento.variables["uwnd"][:] 
v_viento = data_v_viento.variables["vwnd"][:] 
presion = data_presion.variables["pres"][:]
lats = data_u_viento.variables["lat"][:]
lons = data_u_viento.variables["lon"][:]
time = data_u_viento.variables["time"][:]

# Obtenemos array en modo masked_array, necesitamos el array útil para operar
u_viento = u_viento.filled()
v_viento = v_viento.filled()
presion = presion.filled()
lats = lats.filled()
lons = lons.filled()
time = time.filled()

# Cerramos los archivos
data_u_viento.close()
data_v_viento.close()
data_presion.close()

# Transformamos el tiempo a fechas
fechas = num2date(time, units="hours since 1800-01-01 00:00:0.0", calendar="standard")

# Convertimos los objetos cftime a datetime 
fechas = np.array([
    datetime(f.year, f.month, f.day, f.hour, f.minute, f.second)
    for f in fechas
])

# Convertimos Pa a hPa
presion = presion/100 # Conversión de Pa a hPa

# Convertimos las longitudes de 0-360 a -180-180
lons = np.where(lons > 180, lons - 360, lons) # Si la longitud es > 180, restamos 360; si no, la dejamos igual

# Elegimos el punto más cercano a Buenos Aires
lat_BA = -34.6
lon_BA = -58.4

# Índices del punto más cercano
ilat = np.argmin(np.abs(lats - lat_BA)) # abs convierte las diferencias en valores absolutos (distancias) y argmin() nos devuelve el índice del valor mín
ilon = np.argmin(np.abs(lons - lon_BA))

print("La latitud más cercana a Buenos Aires es:", lats[ilat])
print("La longitud más cercana a Buenos Aires es:", lons[ilon])

# Extraemos las series temporales en ese punto
serie_u = u_viento[:, ilat, ilon]
serie_v = v_viento[:, ilat, ilon]
serie_p = presion[:, ilat, ilon] 

# Construimos un DataFrame con fechas, viento y presión 
df_meteo = pd.DataFrame({"Fecha": fechas, "u10": serie_u, "v10": serie_v, "Presión": serie_p})

# Nos interesa la intensidad del viento en m/s
df_meteo["Viento"] = np.sqrt(df_meteo["u10"]**2 + df_meteo["v10"]**2)

# Convertimos la columna fecha a índice (para poder hacer slicing por fechas)
df_meteo = df_meteo.set_index("Fecha")

# Ordenamos ese índice
df_meteo = df_meteo.sort_index()

# Queremos graficar la serie temporal de presión e intensidad del viento desde 2 días antes hasta 2 días después de la ocurrencia de sudestadas
# Recorremos cada evento de la tabla
for i in range(len(tabla_eventos)):

    inicio_evento = tabla_eventos["Inicio"][i]
    fin_evento    = tabla_eventos["Fin"][i]

    # Rango de ±2 días
    inicio_rango = inicio_evento - pd.Timedelta(days=2)
    fin_rango    = fin_evento + pd.Timedelta(days=2)

    # Selección temporal usando el índice datetime
    df_evento = df_meteo.loc[inicio_rango : fin_rango]

    if df_evento.empty:
        print(f"Evento {i}: no hay datos en ese rango")
        continue

    # Generamos la figura como los ejes de los subplots
    fig, ax1 = plt.subplots(figsize=(10, 5))  
    
    # Creamos el eje 1 para la presión
    # Usamos el metodo de ploteo de linea
    ax1.plot(df_evento.index, df_evento["Presión"], color="blue", label="Presión (hPa)")
    # Usamos el metodo para agregar el nombre a los ejes 
    ax1.set_xlabel("Fecha (UTC)", fontweight="bold")
    ax1.set_ylabel("Presión (hPa)", color="black", fontweight="bold")

    # Creamos eje 2 para el viento
    ax2 = ax1.twinx() # Ambos comparten el mismo eje X
    ax2.plot(df_evento.index, df_evento["Viento"], color="purple", linestyle="--", label="Viento (m/s)")
    ax2.set_ylabel("Viento (m/s)", color="black", fontweight="bold")

    # Líneas verticales marcando inicio y fin del evento
    ax1.axvline(inicio_evento, color="#2a557d", linestyle=":")
    ax1.axvline(fin_evento, color="#2a557d", linestyle=":")
    
    # --- Sombra del evento ---
    ax1.axvspan(inicio_evento, fin_evento, 
            color="#2a557d", alpha=0.15, label="_nolegend_")
    
    # --- Grilla suave ---
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Título
    plt.title(
    f"Evento {i+1}: {inicio_evento.strftime('%Y-%m-%d')} a {fin_evento.strftime('%Y-%m-%d')}",
    fontsize=14,
    fontweight="bold",
    style="italic",
    pad=12,
    backgroundcolor="#7facc9"
    )
    
    # Combinamos leyendas de ax1 y ax2
    lineas1, labels1 = ax1.get_legend_handles_labels()
    lineas2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lineas1 + lineas2, labels1 + labels2,
           loc="upper left", frameon=True, framealpha=0.8)
    
    plt.tight_layout(pad=2.0)
    fig.savefig(f"Evento_{i+1}.png", dpi=300, bbox_inches="tight")
    plt.show()
    
# Realizaremos un mapa con la composición del campo de presión superpuesto al campo de viento para la hora más cercana al máximo de cada evento, 12 horas antes y 12 horas después, en un panel de 3 columnas

# Armado del reticulado con las latitudes y longitudes
lon_grid, lat_grid = np.meshgrid(lons,lats)

# Nuevamente utilizamos un ciclo para recorrer los eventos y obtener la fecha del máximo de altura alcanzada
# Creamos una lista vacía donde se guardarán los datos
fechas_max_eventos = []

for i in range(len(tabla_eventos)):
    
    inicio = tabla_eventos.loc[i, "Inicio"]
    fin = tabla_eventos.loc[i, "Fin"]
    alt_max = tabla_eventos.loc[i, "Altura máxima alcanzada"]
    
    # Nos concentramos en el intervalo del evento
    df_evento = df[(df["fecha"] >= inicio) & (df["fecha"] <= fin)]
    
    # Buscamos la fila donde se alcanza la mayor altura. Y usamos [0] para quedarnos con la primera fila que registra ese valor
    fila_max = df_evento[df_evento["Altura (m)"] == alt_max].iloc[0]
    
    # Nos interesa guardar el dato de la fecha
    fecha_max_evento = fila_max["fecha"]
    
    fechas_max_eventos.append(fecha_max_evento)
    
# Debemos guardar la información de los 43 eventos que promediaremos
n_eventos = len(fechas_max_eventos)

u_12hs_antes = []
u_max = []
u_12hs_desp = []
v_12hs_antes = []
v_max = []
v_12hs_desp = []
pres_12hs_antes = []
pres_max = []
pres_12hs_desp = []

# Definimos una función
def búsqueda_idx_fecha(fechas, fecha_objetivo):
    """
    Dada una serie de fechas provenientes del NetCDF, y una fecha objetivo 
    (por ejemplo, la fecha del máximo de altura del rio alcanzada), el 
    propósito de esta función es devolver el índice del NetCDF correspondiente 
    a la fecha más cercana a la ingresada.
    
    Parámetros:
    fechas : np.array de datetime
        Array con las fechas provenientes del NetCDF.
    fecha_objetivo : datetime
        Fecha del evento para la cual buscamos el índice temporal más cercano.
    """
    diferencia = np.abs(fechas - fecha_objetivo)
    return np.argmin(diferencia)

# Extraemos los campos para cada evento
for fecha_max_evento in fechas_max_eventos:
    
    # Para poder hacer la diferencia en la función, es necesario que las fechas sean del mismo tipo
    fecha_max_evento = fecha_max_evento.to_pydatetime()
    
    # Colocamos las fechas objetivos que usaremos en nuestra función
    fechas_objetivo = [fecha_max_evento - timedelta(hours=12), fecha_max_evento, fecha_max_evento + timedelta(hours=12)]
    
    # Utilizamos la función que creamos para buscar el indice para los tres tiempos que nos interesan
    idx_12hs_antes = búsqueda_idx_fecha(fechas, fechas_objetivo[0]) 
    idx_max = búsqueda_idx_fecha(fechas, fechas_objetivo[1])
    idx_12hs_desp = búsqueda_idx_fecha(fechas, fechas_objetivo[2])
    
    # Una vez que conocemos esos índices, extraemos de los campos esos tiempos
    u_12hs_antes.append(u_viento[idx_12hs_antes, :, :])
    u_max.append(u_viento[idx_max, :, :])
    u_12hs_desp.append(u_viento[idx_12hs_desp, :, :])
    
    v_12hs_antes.append(v_viento[idx_12hs_antes, :, :])
    v_max.append(v_viento[idx_max, :, :])
    v_12hs_desp.append(v_viento[idx_12hs_desp, :, :])
    
    pres_12hs_antes.append(presion[idx_12hs_antes, :, :])
    pres_max.append(presion[idx_max, :, :])
    pres_12hs_desp.append(presion[idx_12hs_desp, :, :])
    
# Convertimos las listas a arrays 3D: asi tenemos la información como (evento, lat, lon) y luego podemos hacer el promedio
u_12hs_antes = np.array(u_12hs_antes)
u_max = np.array(u_max)
u_12hs_desp = np.array(u_12hs_desp)
v_12hs_antes = np.array(v_12hs_antes)
v_max = np.array(v_max)
v_12hs_desp = np.array(v_12hs_desp)
pres_12hs_antes = np.array(pres_12hs_antes)
pres_max = np.array(pres_max)
pres_12hs_desp = np.array(pres_12hs_desp)

# Promedios por evento: promedio de 43 mapas
u_prom_12a = u_12hs_antes.mean(axis=0)
u_prom_max = u_max.mean(axis=0)
u_prom_12d = u_12hs_desp.mean(axis=0)

v_prom_12a = v_12hs_antes.mean(axis=0)
v_prom_max = v_max.mean(axis=0)
v_prom_12d = v_12hs_desp.mean(axis=0)  

pres_prom_12a = pres_12hs_antes.mean(axis=0)
pres_prom_max = pres_max.mean(axis=0)
pres_prom_12d = pres_12hs_desp.mean(axis=0)  

# Agrupamos los campos en listas 
campos_u = [u_prom_12a, u_prom_max, u_prom_12d]
campos_v = [v_prom_12a, v_prom_max, v_prom_12d]
campos_pres = [pres_prom_12a, pres_prom_max, pres_prom_12d]

# Graficado 
fig, axs = plt.subplots(
    nrows=1, ncols=3, figsize=(15, 6),
    subplot_kw={'projection': ccrs.PlateCarree()}
)

titulos = ["12 h antes", "Máximo del evento", "12 h después"]

axs = axs.flatten()

niveles = np.arange(990, 1032, 4)  # 990–1030 hPa cada 4 hPa

for i in range(3):

    ax = axs[i]

    # Mapa base
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle='-', alpha=0.5)
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            'cultural', 'admin_1_states_provinces_lines',
            '10m', facecolor='none'
        ),
        edgecolor='black', linewidth=0.5
    )

    # Campo de presión
    im = ax.contourf(
        lon_grid, lat_grid, campos_pres[i],
        levels=niveles, cmap="coolwarm", extend="both",
        transform=ccrs.PlateCarree()
    )
    
    # Isobaras
    ax.contour(
    lon_grid, lat_grid, campos_pres[i],
    levels=niveles, colors='k', linewidths=0.3,
    transform=ccrs.PlateCarree()
    )

    # Campo de viento
    ax.quiver(
        lon_grid[:, :], lat_grid[:, :],
        campos_u[i][:, :], campos_v[i][:, :],
        transform=ccrs.PlateCarree(),
        scale=80, scale_units='height', color='black'
    )

    # Grilla
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                      alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

    ax.set_title(titulos[i], fontsize=12)

# Colorbar
cbar_ax = fig.add_axes([0.25, 0.10, 0.5, 0.025])  
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
cbar.set_label("Presión (hPa)", fontsize=11)

# Título de la figura 
fig.suptitle("Composición promedio (43 eventos): Presión + Viento",
             fontsize=16, y=0.98, fontweight="bold", style="italic")

plt.subplots_adjust(
    top=0.88,     
    bottom=0.18,  
    wspace=0.28   
)

fig.savefig("Comp_presión_viento.png", dpi=300, bbox_inches='tight')
plt.show()


# Dado lo obtenido en el gráfico para todo el dominio, buscamos mostrar que esta sucediendo, con pcolormes podemos apreciar un gradiente espacial grande
fig, axs = plt.subplots(
    nrows=1, ncols=3, figsize=(15, 6),
    subplot_kw={'projection': ccrs.PlateCarree()}
)

titulos = ["12 h antes", "Máximo del evento", "12 h después"]

axs = axs.flatten()

niveles = np.arange(990, 1032, 4)  # 990–1030 hPa cada 4 hPa

cm = None # Lo definimos de esta manera para que dentro del ciclo, no se genere una colorbar para cada figura. Solo queremos visualizar una horizontalmente 
for i in range(3):

    ax = axs[i]

    # Mapa base
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle='-', alpha=0.5)
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            'cultural', 'admin_1_states_provinces_lines',
            '10m', facecolor='none'
        ),
        edgecolor='black', linewidth=0.5
    )
 
    # Campo de presión
    cm = ax.pcolormesh(lon_grid, lat_grid, campos_pres[i][:, :], transform=ccrs.PlateCarree(), vmin=900)


    # Campo de viento
    ax.quiver(
        lon_grid[:, :], lat_grid[:, :],
        campos_u[i][:, :], campos_v[i][:, :],
        transform=ccrs.PlateCarree(),
        scale=250, scale_units='height', color='black'
    )

    # Grilla
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                      alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

    ax.set_title(titulos[i], fontsize=12)
    
# Colorbar
cbar_ax = fig.add_axes([0.25, 0.10, 0.5, 0.025])  
cbar = fig.colorbar(cm, cax=cbar_ax, orientation='horizontal')
cbar.set_label("Presión (hPa)", fontsize=11)

# Título de la figura 
fig.suptitle("Gradiente barométrico en los campos",
             fontsize=16, y=0.98, fontweight="bold", style="italic")

plt.subplots_adjust(
    top=0.88,     
    bottom=0.18,  
    wspace=0.28   
)

fig.savefig("Gradiente_Comp_presión_viento.png", dpi=300, bbox_inches='tight')
plt.show()

# Decidimos graficar para un dominio regional en particular 
lat_min, lat_max = -45, -30
lon_min, lon_max = -65, -50

lat_mask = (lats >= lat_min) & (lats <= lat_max)
lon_mask = (lons >= lon_min) & (lons <= lon_max)

# Subconjuntos de coordenadas
lats_sub = lats[lat_mask]
lons_sub = lons[lon_mask]

# Nueva grilla recortada
lon_grid_sub, lat_grid_sub = np.meshgrid(lons_sub, lats_sub)

# Recortamos los campos promediados
u_prom_12a_sub = u_prom_12a[np.ix_(lat_mask, lon_mask)]
u_prom_max_sub = u_prom_max[np.ix_(lat_mask, lon_mask)]
u_prom_12d_sub = u_prom_12d[np.ix_(lat_mask, lon_mask)]

v_prom_12a_sub = v_prom_12a[np.ix_(lat_mask, lon_mask)]
v_prom_max_sub = v_prom_max[np.ix_(lat_mask, lon_mask)]
v_prom_12d_sub = v_prom_12d[np.ix_(lat_mask, lon_mask)]
 
pres_prom_12a_sub = pres_prom_12a[np.ix_(lat_mask, lon_mask)] 
pres_prom_max_sub = pres_prom_max[np.ix_(lat_mask, lon_mask)]
pres_prom_12d_sub = pres_prom_12d[np.ix_(lat_mask, lon_mask)] 

# Agrupamos los campos en listas 
campos_u_sub = [u_prom_12a_sub, u_prom_max_sub, u_prom_12d_sub]
campos_v_sub = [v_prom_12a_sub, v_prom_max_sub, v_prom_12d_sub]
campos_pres_sub = [pres_prom_12a_sub, pres_prom_max_sub, pres_prom_12d_sub]

# Graficado 
fig, axs = plt.subplots(
    nrows=1, ncols=3, figsize=(15, 6),
    subplot_kw={'projection': ccrs.PlateCarree()}
)

titulos = ["12 h antes", "Máximo del evento", "12 h después"]

axs = axs.flatten()

niveles = np.arange(990, 1032, 4)  # 990–1030 hPa cada 4 hPa

for i in range(3):

    ax = axs[i]

    # Mapa base
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle='-', alpha=0.5)
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            'cultural', 'admin_1_states_provinces_lines',
            '10m', facecolor='none'
        ),
        edgecolor='black', linewidth=0.5
    )

    # Campo de presión
    im = ax.contourf(
        lon_grid_sub, lat_grid_sub, campos_pres_sub[i],
        levels=niveles, cmap="coolwarm", extend="both",
        transform=ccrs.PlateCarree()
    )
    
    # Isobaras
    ax.contour(
    lon_grid_sub, lat_grid_sub, campos_pres_sub[i],
    levels=niveles, colors='k', linewidths=0.3,
    transform=ccrs.PlateCarree()
    )

    # Campo de viento
    ax.quiver(
        lon_grid_sub[:, :], lat_grid_sub[:, :],
        campos_u_sub[i][:, :], campos_v_sub[i][:, :],
        transform=ccrs.PlateCarree(),
        scale=80, scale_units='height', color='black'
    )

    # Grilla
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                      alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

    ax.set_title(titulos[i], fontsize=12)

# Colorbar
cbar_ax = fig.add_axes([0.25, 0.14, 0.5, 0.025])  
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
cbar.set_label("Presión (hPa)", fontsize=11)

# Título de la figura 
fig.suptitle("Composición promedio (43 eventos): Presión + Viento",
             fontsize=16, y=0.92, fontweight="bold", style="italic")

plt.subplots_adjust(
    top=0.88,     
    bottom=0.18,  
    wspace=0.28   
)

fig.savefig("Comp_dom_recortado.png", dpi=300, bbox_inches='tight')
plt.show()
