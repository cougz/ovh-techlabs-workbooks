# OVHcloud TechLabs - Workbooks

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A comprehensive collection of workbooks for OVHcloud TechLabs.

## Project Status

**Status**: Active maintenance - Example project  
**Support Level**: Community support  
**Type**: Educational workbooks and documentation

> **Note**: This is not an official OVHcloud product. This repository contains educational workbooks and examples for learning OVHcloud services.

## Quick Links

- **Documentation Site**: [https://cougz.github.io/ovh-techlabs-workbooks/](https://cougz.github.io/ovh-techlabs-workbooks/)
- **Source Repository**: [https://github.com/cougz/ovh-techlabs-workbooks](https://github.com/cougz/ovh-techlabs-workbooks)

## Available Workbooks

### Public Cloud

#### AI Endpoints
- **RAG Workbook** - Build lab-ready RAG systems using OVHcloud AI Endpoints
- **VLM Workbook** - Car damage verification using Vision Language Models

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
│   ├── public-cloud/       # Public Cloud documentation
│   └── stylesheets/        # Custom CSS
├── mkdocs.yml             # MkDocs configuration
├── requirements.txt       # Python dependencies
├── .github/workflows/     # GitHub Actions for deployment
├── AUTHORS                # Copyright holders
├── CONTRIBUTING.md        # Contribution guidelines
├── CONTRIBUTORS           # List of contributors
├── LICENSE                # Apache 2.0 License
├── LICENSES/              # License information
└── MAINTAINERS           # Project maintainers
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Code of Conduct

Please note that this project follows the OVHcloud code of conduct. By participating, you are expected to uphold this code.

## Support

This is a community-supported project. For questions and issues:
- Create an issue in this repository
- Check existing documentation
- Review closed issues for solutions

## Disclaimer

This repository is not an official OVHcloud product and is maintained by the community for educational purposes.
