# Installer Integration with SUMO Installation

## Overview

Yes, the executable installer can automatically install SUMO and set SUMO_HOME. Here are the options:

## Option 1: Bundle SUMO in Installer (Recommended for Self-Contained Distribution)

### Approach:
- Include SUMO installation files in the installer package
- Installer extracts SUMO to a standard location
- Automatically sets SUMO_HOME in the application

### Pros:
- ✅ Self-contained - users don't need to install SUMO separately
- ✅ Works offline
- ✅ Consistent SUMO version
- ✅ Better user experience

### Cons:
- ❌ Large installer size (500MB - 1GB+)
- ❌ Platform-specific installers needed (Windows/macOS/Linux)
- ❌ Need to update installer when SUMO version changes

### Implementation:
1. **Download SUMO** for each platform during build
2. **Bundle SUMO** in installer package
3. **Extract SUMO** during installation to `{AppData}/SUMO` or `{ProgramFiles}/SUMO`
4. **Set SUMO_HOME** automatically in app config
5. **Create shortcuts** and registry entries

## Option 2: Download SUMO During Installation

### Approach:
- Installer downloads SUMO from official sources during installation
- Extracts and installs SUMO
- Sets SUMO_HOME automatically

### Pros:
- ✅ Smaller initial installer
- ✅ Always gets latest SUMO version
- ✅ Can check for updates

### Cons:
- ❌ Requires internet connection during installation
- ❌ Slower installation
- ❌ May fail if download fails

## Option 3: First-Run Setup Wizard

### Approach:
- Installer installs only the application
- First run shows setup wizard
- Wizard can:
  - Detect existing SUMO installation
  - Download/install SUMO if needed
  - Set SUMO_HOME

### Pros:
- ✅ Small installer
- ✅ Flexible - users can use existing SUMO or install new
- ✅ Can skip if SUMO already installed

### Cons:
- ❌ Requires user interaction
- ❌ More complex implementation

## Recommended Implementation: Option 1 (Bundle SUMO)

### Installer Tools:

#### Windows:
- **Inno Setup** (Recommended)
  - Free, open-source
  - Good for bundling files
  - Can run scripts during installation
  - Can set registry entries

- **NSIS** (Nullsoft Scriptable Install System)
  - Free, open-source
  - Very flexible
  - Can download files during install

- **WiX Toolset**
  - Microsoft's installer tool
  - More complex but professional

#### macOS:
- **macOS Installer Package** (.pkg)
  - Built-in tool: `pkgbuild` and `productbuild`
  - Can bundle SUMO and set paths

#### Linux:
- **AppImage** (Recommended for portability)
  - Single file, no installation needed
  - Can bundle SUMO inside

- **Debian Package** (.deb)
  - For Debian/Ubuntu systems
  - Can have SUMO as dependency or bundle it

- **RPM Package** (.rpm)
  - For Red Hat/CentOS systems

### Implementation Steps:

1. **Build Script**:
   ```bash
   # Download SUMO for each platform
   # Extract to installer package
   # Create installer with bundled SUMO
   ```

2. **Installer Script** (Inno Setup example):
   ```pascal
   [Files]
   Source: "sumo\*"; DestDir: "{app}\sumo"; Flags: recursesubdirs
   
   [Code]
   procedure InitializeWizard();
   begin
     // Set SUMO_HOME during installation
   end;
   ```

3. **Application Code**:
   - Check for SUMO in installation directory
   - Auto-detect and set SUMO_HOME on first run
   - Fall back to user-configured path if not found

## Code Changes Needed

### 1. Auto-Detection on First Run:
```python
def auto_detect_sumo_home():
    """Auto-detect SUMO_HOME from common locations."""
    # Check installation directory
    app_dir = Path(sys.executable).parent
    sumo_dir = app_dir / 'sumo'
    if (sumo_dir / 'bin').exists():
        return str(sumo_dir)
    
    # Check environment variable
    if os.environ.get('SUMO_HOME'):
        return os.environ['SUMO_HOME']
    
    # Check common installation paths
    # ... (existing detection logic)
    
    return None
```

### 2. First-Run Setup:
- Check if SUMO_HOME is set
- If not, try auto-detection
- If still not found, prompt user to set it
- Show helpful message with download link

## File Structure for Installer

```
installer-package/
├── app/                    # Your application
│   ├── executable.exe
│   └── ...
├── sumo/                   # Bundled SUMO
│   ├── bin/
│   ├── tools/
│   └── ...
└── installer-script.iss    # Inno Setup script
```

## Size Considerations

- **Application**: ~50-100MB (with PySide6)
- **SUMO**: ~100-500MB (platform-specific)
- **Total Installer**: ~150-600MB

This is acceptable for modern applications (similar to games, IDEs, etc.)

## Recommendation

**Use Option 1 (Bundle SUMO)** with:
- **Windows**: Inno Setup
- **macOS**: macOS Installer Package
- **Linux**: AppImage or platform-specific packages

This provides the best user experience - users just run the installer and everything works.

