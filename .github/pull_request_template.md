# Pull Request

## Description

Brief description of what this PR does and why.

## Type of Change

- [ ] 🐛 Bug fix (non-breaking change which fixes an issue)
- [ ] ✨ New feature (non-breaking change which adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] 📚 Documentation update
- [ ] 🔧 Refactoring (no functional changes)
- [ ] ⚡ Performance improvement
- [ ] 🧪 Test improvements

## Related Issues

Fixes #(issue number)
Closes #(issue number)
Related to #(issue number)

## Changes Made

- [ ] Change 1
- [ ] Change 2
- [ ] Change 3

## Testing

### How Has This Been Tested?

- [ ] Unit tests
- [ ] Integration tests
- [ ] Manual testing

### Test Configuration

- OS: [e.g. Ubuntu 22.04]
- Python: [e.g. 3.10.12]
- GPUs: [e.g. 4x Tesla M10]
- Models tested: [e.g. llama3.1:8b, phi4:latest]

### Test Commands

```bash
# Commands used to test this change
python test_context_fix.py
curl http://localhost:11435/health
```

## API Changes

### New Endpoints

- `POST /api/new-endpoint` - Description

### Modified Endpoints

- `GET /api/existing` - What changed

### Breaking Changes

- [ ] No breaking changes
- [ ] Breaking changes (describe migration path below)

**Migration Path:**
```bash
# Steps users need to take to migrate
```

## Configuration Changes

- [ ] No configuration changes
- [ ] New configuration options (backward compatible)
- [ ] Modified configuration (breaking change)

**New/Modified Config:**
```yaml
# Example configuration changes
new_option: default_value
```

## Documentation

- [ ] README.adoc updated
- [ ] API documentation updated
- [ ] Code comments added/updated
- [ ] CHANGELOG.md updated (for releases)

## Performance Impact

- [ ] No performance impact expected
- [ ] Performance improvement
- [ ] Potential performance regression (explain below)

**Performance Notes:**
Describe any performance implications, memory usage changes, or GPU impact.

## Security Considerations

- [ ] No security implications
- [ ] Security improvement
- [ ] Potential security impact (explain below)

## Deployment Impact

- [ ] No deployment changes required
- [ ] Service restart required
- [ ] Configuration migration required
- [ ] Database/storage migration required

## Checklist

### Code Quality

- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] My changes generate no new warnings or errors

### Testing

- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] I have tested the changes with real model inference

### Documentation

- [ ] I have made corresponding changes to the documentation
- [ ] I have updated the README.adoc if needed
- [ ] I have added docstrings to new functions/classes

### Compatibility

- [ ] My changes are backward compatible
- [ ] I have tested with multiple API formats (OpenAI, Ollama, etc.)
- [ ] I have considered multi-GPU scenarios

## Screenshots

If applicable, add screenshots to help explain your changes (especially for dashboard/UI changes).

## Additional Notes

Any additional information, considerations, or context that reviewers should know about.

## Reviewer Notes

Areas that need special attention during review:

- [ ] Performance implications
- [ ] Security considerations  
- [ ] API compatibility
- [ ] Error handling
- [ ] Configuration management