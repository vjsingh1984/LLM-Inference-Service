# Contributing to LLM Inference Service

We welcome contributions to the LLM Inference Service! This document provides guidelines for contributing to the project.

## 🚀 Quick Start for Contributors

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/llm-inference-service.git
   cd llm-inference-service
   ```
3. **Set up development environment**:
   ```bash
   pip install -r requirements.txt
   npm install  # For diagram generation tools
   ```
4. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 📋 Development Guidelines

### Code Style

- **Python**: Follow PEP 8 with 100-character line limit
- **Type hints**: Required for all public interfaces and method signatures
- **Docstrings**: Use Google-style docstrings for all classes and public methods
- **Imports**: Organize using isort (`isort --profile black .`)
- **Comments**: Write clear, concise comments explaining the "why", not the "what"

### Code Organization

The project follows a modular architecture:

```
ollama_server/
├── core/              # Core functionality (schemas, tracking, execution)
├── models/            # Model management and discovery
├── adapters/          # API format adapters (OpenAI, Ollama, etc.)
├── api/               # Web API routes and handlers
├── utils/             # Utilities (logging, model inspection)
├── config.py          # Configuration management
└── main.py            # Application entry point
```

### Testing

- **Write tests** for new features and bug fixes
- **Run tests** before submitting: `python -m pytest tests/`
- **Test coverage**: Aim for >80% coverage on new code
- **Integration tests**: Test actual API endpoints when possible

### Documentation

- **Update README.adoc** for new features or API changes
- **Update docstrings** for any modified public interfaces
- **Diagram updates**: Update `.mmd` files if architecture changes
- **API examples**: Include working curl examples for new endpoints

## 🛠️ Common Development Tasks

### Adding a New API Format

1. Create adapter in `ollama_server/adapters/your_format.py`
2. Inherit from `RequestAdapter` base class
3. Implement `parse_request()` and `format_response()` methods
4. Add routes in `ollama_server/api/routes.py`
5. Update tests and documentation

### Adding Model Support

1. Update model detection in `ModelManager.get_model_family()`
2. Add context size patterns in `ModelManager._detect_context_size()`
3. Update `ModelInspector` if special handling needed
4. Test with actual model files

### Performance Improvements

1. Profile with `cProfile` or similar tools
2. Focus on inference path optimizations
3. Consider GPU memory usage impacts
4. Test with multiple concurrent requests

## 🐛 Bug Reports

When reporting bugs, please include:

- **Environment**: OS, Python version, GPU info
- **Steps to reproduce**: Clear, minimal reproduction steps
- **Expected vs actual behavior**
- **Logs**: Relevant log snippets (with sensitive info removed)
- **Configuration**: Relevant parts of your config

## ✨ Feature Requests

For feature requests:

- **Use case**: Describe the problem you're trying to solve
- **Proposed solution**: How you envision the feature working
- **Alternatives**: Other approaches you've considered
- **Compatibility**: Impact on existing APIs and configurations

## 🔍 Code Review Process

1. **All changes** require pull request review
2. **CI checks** must pass (tests, linting, type checking)
3. **Documentation** updates required for user-facing changes
4. **Breaking changes** require discussion and migration path
5. **Performance** impact should be considered and tested

### Pull Request Guidelines

- **Clear title**: Summarize the change in 50 characters or less
- **Description**: Explain what changed and why
- **Testing**: Describe how you tested the changes
- **Screenshots**: Include for UI/dashboard changes
- **Breaking changes**: Clearly mark and explain migration path

## 🚀 Release Process

1. **Version bumping**: Follow semantic versioning (MAJOR.MINOR.PATCH)
2. **Changelog**: Update with all user-facing changes
3. **Testing**: Thorough testing on different configurations
4. **Documentation**: Ensure all docs are up to date
5. **Tag release**: Create Git tag and GitHub release

## 📚 Architecture Decisions

For significant architectural changes:

1. **Create issue** discussing the change
2. **Design document** for complex changes
3. **Community feedback** before implementation
4. **Backward compatibility** plan when possible

## 🤝 Community Guidelines

- **Be respectful** and inclusive in all communications
- **Help others** learn and contribute
- **Ask questions** - no question is too basic
- **Share knowledge** through good documentation and examples
- **Report issues** constructively with actionable information

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and community support
- **Documentation**: Check README.adoc and inline code docs
- **Examples**: Look at test files for usage patterns

## 🏷️ Labels and Issues

We use these labels to organize issues:

- `bug`: Something isn't working correctly
- `enhancement`: New features or improvements
- `documentation`: Documentation improvements
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention needed
- `performance`: Performance-related improvements
- `api`: API compatibility or new endpoints
- `gpu`: GPU-related functionality

## 📄 License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

Thank you for contributing to the LLM Inference Service! 🎉