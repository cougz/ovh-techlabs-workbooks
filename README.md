# OVHcloud TechLabs - Workbooks and Tutorials

A comprehensive collection of tutorials and workbooks for OVHcloud TechLabs.

## Quick Links

- **Documentation Site**: [https://cougz.github.io/ovh-techlabs-workbooks/](https://cougz.github.io/ovh-techlabs-workbooks/)
- **Source Repository**: [https://github.com/cougz/ovh-techlabs-workbooks](https://github.com/cougz/ovh-techlabs-workbooks)

## Available Workbooks

### Public Cloud

#### AI Endpoints
- **RAG Tutorial** - Build lab-ready RAG systems using OVHcloud AI Endpoints
- **VLM Tutorial** - Car damage verification using Vision Language Models

## Development

This repository uses MkDocs with the Material theme to generate documentation. To run locally:

```bash
pip install -r requirements.txt
mkdocs serve
```

## Deployment

The documentation is automatically deployed to GitHub Pages using GitHub Actions when changes are pushed to the main branch. The workflow builds the MkDocs site and publishes it to `https://cougz.github.io/ovh-techlabs-workbooks/`.

## Repository Structure

```
ovh-techlabs-workbooks/
├── docs/                    # MkDocs documentation source
│   ├── assets/             # Images, icons, and static files
│   ├── en/                 # English documentation
│   ├── public-cloud/       # Public Cloud documentation
│   └── stylesheets/        # Custom CSS
├── public-cloud/           # Tutorial implementations and code
│   └── ai-endpoints/       # AI Endpoints tutorials
├── mkdocs.yml             # MkDocs configuration
├── requirements.txt       # Python dependencies
└── .github/workflows/     # GitHub Actions for deployment
```

## Contributing

Contributions are welcome! Please feel free to create issues or submit a Pull Request.
