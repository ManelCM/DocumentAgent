## DocumentAgent - OCR Multimodal con LangGraph

Pipeline estructural para documentos complejos:

1. Ingesta PDF/imagen.
2. Deteccion de layouts.
3. Clasificacion por tipo de bloque.
4. Reading order.
5. Jerarquia padre-hijo.
6. Enrutamiento por nodos especializados.
7. Salida JSON estructurada para downstream.

### Estructura

- `src/document_agent/graph.py`: grafo LangGraph.
- `src/document_agent/nodes.py`: nodos de procesamiento.
- `src/document_agent/layout.py`: mapeo de labels y deteccion.
- `src/document_agent/order.py`: orden de lectura.
- `src/document_agent/hierarchy.py`: relaciones jerarquicas.
- `src/document_agent/cli.py`: ejecucion por linea de comandos.

### Ejecucion

```bash
python -m document_agent --input "data/paper1.pdf" --output "outputs/paper1.json"
```

Si usas el layout de carpetas actual:

```bash
cd DocumentAgent
$env:PYTHONPATH="src"
python -m document_agent --input "data/paper1.pdf" --output "outputs/paper1.json"
```

### Entorno recomendado (aislado)

PowerShell:

```powershell
cd DocumentAgent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### OpenAI API (VLM)

Configura tu clave en variables de entorno o con `.env`:

```powershell
cd DocumentAgent
Copy-Item .env.example .env
```

Variables:

- `OPENAI_API_KEY`: clave de OpenAI.
- `OPENAI_VLM_IMAGE_MODEL`: modelo para `image_node`.
- `OPENAI_VLM_CHART_MODEL`: modelo para `chart_node`.
- `OPENAI_VLM_FORMULA_MODEL`: modelo para `formula_node`.

Prompts actuales:

- `image_node`: descripcion objetiva de la figura y texto visible en formato JSON.
- `chart_node`: tipo de grafica, ejes, tendencias, extremos y takeaway en JSON.
- `formula_node`: extraccion a LaTeX y significado breve en JSON.

### Nodos especializados

- `text_node`: OCR de regiones de texto.
- `image_node`: descripcion VLM con OpenAI (con fallback).
- `chart_node`: interpretacion semantica con OpenAI (con fallback).
- `formula_node`: extraccion/formato matematico con OpenAI (con fallback).
- `association_node`: vinculacion figura-caption.

### Salida

JSON con:

- bloques ordenados por lectura.
- tipo de bloque.
- bbox.
- jerarquia (`parent_id`, `child_ids`).
- relaciones semanticas (`relations`).
- payload por nodo especialista.
