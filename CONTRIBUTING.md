# Contributing to OVHcloud TechLabs Workbooks

Thank you for your interest in contributing to OVHcloud TechLabs Workbooks! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you are expected to uphold our Code of Conduct:
- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on what is best for the community
- Show empathy towards other community members

## How to Contribute

### Reporting Issues

- Check if the issue has already been reported
- Use the issue templates when available
- Provide clear and detailed information about the issue
- Include steps to reproduce when reporting bugs

### Submitting Changes

1. **Fork the repository**
   - Create your own fork of the repository
   - Clone it locally: `git clone https://github.com/YOUR_USERNAME/ovh-techlabs-workbooks.git`

2. **Create a branch**
   - Create a new branch for your changes: `git checkout -b feature/your-feature-name`
   - Use descriptive branch names

3. **Make your changes**
   - Follow the existing code style and conventions
   - Write clear, concise commit messages
   - Keep commits focused and atomic

4. **Test your changes**
   - Ensure all workbooks still function correctly
   - Test any new content thoroughly
   - Verify documentation builds correctly with MkDocs

5. **Submit a Pull Request**
   - Push your changes to your fork
   - Create a pull request with a clear title and description
   - Link any related issues

### Workbook Guidelines

When contributing new workbooks or modifying existing ones:

- Use clear, educational language
- Include practical, working examples
- Follow the established workbook structure
- Test all code examples thoroughly
- Keep content focused on OVHcloud services
- Use the term "workbook" consistently (not "tutorial" or "guide")

### Documentation Style

- Use Markdown for all documentation
- Follow the Material for MkDocs conventions
- Include code highlighting for examples
- Add appropriate warnings and tips using admonitions
- Keep paragraphs concise and scannable

### Commit Message Format

Use clear and descriptive commit messages:
```
<type>: <subject>

<body>

<footer>
```

Types:
- `feat`: New feature or workbook
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Formatting changes
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

## Development Setup

1. Install Python 3.8+
2. Install dependencies: `pip install -r requirements.txt`
3. Run locally: `mkdocs serve`
4. View at: `http://localhost:8000`

## Questions?

If you have questions, please:
- Check existing documentation
- Search through issues
- Create a new issue with the question label

## License

By contributing to OVHcloud TechLabs Workbooks, you agree that your contributions will be licensed under the Apache License 2.0.