# GitHub Setup Instructions

The titus-core package is ready to push to GitHub.

## Steps to Complete

### 1. Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `titus-core`
3. Visibility: **Private**
4. Do NOT initialize with README (we already have one)
5. Click "Create repository"

### 2. Push to GitHub

```bash
cd /Users/leeclarke/repos/titus-core

# Add GitHub as remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/titus-core.git

# Push code
git push -u origin main

# Push tag
git push origin v0.1.0
```

### 3. Verify

Check that https://github.com/YOUR_USERNAME/titus-core shows:
- All code files
- Tag v0.1.0 in releases

### 4. Test Installation from GitHub

```bash
# In a new directory:
pip install git+https://github.com/YOUR_USERNAME/titus-core.git@v0.1.0
```

---

## Repository is Ready

All code is committed and tagged. Just needs GitHub remote setup!

Current status:
- ✅ Code committed
- ✅ Tagged as v0.1.0
- ⏸️  Awaiting GitHub repo creation
- ⏸️  Then: git push

After pushing, you can install via:
```
pip install git+https://github.com/YOUR_USERNAME/titus-core.git@v0.1.0
```

