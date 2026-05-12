# Proyecto Integrador — Teoría del Riesgo

> Solución de referencia del proyecto integrador del curso *Python para Desarrollo de APIs e Inteligencia Artificial* (Universidad Santo Tomás · pregrado en Estadística · 2026-I).

Este repositorio contiene la **implementación de referencia** que el docente construye en paralelo con los estudiantes. Sirve como ejemplo, no como plantilla a forkear: cada estudiante construye su propia versión desde cero, inspirándose en la estructura y decisiones aquí registradas.

## ¿Qué resuelve?

Una API en FastAPI que aplica conceptos de **Teoría del Riesgo** (modelo, validación, persistencia, despliegue). Articula los módulos del curso:

- **M4 / M7** — Validación de entrada con Pydantic
- **M6** — API REST con FastAPI
- **M8** — Inyección de dependencias y configuración
- **M9** — Persistencia con SQLAlchemy
- **M10** — Pruebas con pytest
- **M11** — Contenedorización con Docker
- **M12** — CI/CD con GitHub Actions
- **M13** — Modelo ML servido en producción

## Estado

🚧 **Scaffolding inicial.** El código se desarrolla en clase a partir de la Clase II del proyecto integrador.

## Relación con el curso

- Repo del curso: <https://github.com/JotaMao1985/PYTHON-PARA-DESARROLLO-DE-APIS-E-INTELIGENCIA-ARTIFICIAL-20261>
- Este repo es **submódulo** del repo del curso, anclado en su carpeta `Proyecto_I/`.
- Para clonar el curso con este submódulo incluido:

```bash
git clone --recurse-submodules https://github.com/JotaMao1985/PYTHON-PARA-DESARROLLO-DE-APIS-E-INTELIGENCIA-ARTIFICIAL-20261.git
```

- Si ya lo clonaste sin submódulos:

```bash
git submodule update --init --recursive
```

## Demos pedagógicas (no aquí)

Los snippets cortos que se proyectan en clase (CRUD, Hello World, Bernoulli, generador de clases con IA, etc.) viven en la carpeta `Examples/` del repo del curso, **no en este repositorio**.

## Docente

**Javier Mauricio Sierra** — Universidad Santo Tomás
