# Trabajo Final Laboratorio de Procesamiento de Información Meteorológica - DCAO UBA: Análisis de Sudestadas en Buenos Aires

En el marco del trabajo final de la materia, se busca identificar y analizar eventos de sudestada en el Río de la Plata utilizando datos hidrométricos y campos de presión y viento provenientes de reanálisis.

El análisis se realiza mediante un script en Python que automatiza la detección de eventos, la extracción de datos de reanálisis y la generación de figuras.

---

## Objetivos del análisis

1. **Identificar eventos de sudestada**  
   Se consideran sudestadas aquellos episodios donde la altura del río supera 
   los **2.5 m**.  
   Para cada evento se calcula automáticamente:
   - Fecha de inicio  
   - Fecha de fin  
   - Duración (días y horas)  
   - Altura máxima alcanzada  

2. **Analizar la evolución temporal de presión y viento**  
   A partir de los datos de presión y viento se selecciona el punto de grilla más 
   cercano a Buenos Aires.  
   Para cada evento se grafica:
   - Serie temporal de presión (hPa)  
   - Serie temporal de intensidad del viento (m/s)  
   desde **48 h antes** hasta **48 h después** del máximo del evento.

3. **Generar composiciones espaciales de presión y viento**  
   Para cada evento se extraen 3 tiempos:
   - 12 horas antes del máximo  
   - Hora del máximo  
   - 12 horas después  
