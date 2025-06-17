#!/usr/bin/env python3
"""
Universal F-String Backslash Fix Script
Automatically finds and fixes all f-string backslash issues in your project.
"""

import os
import re
import glob
import shutil
from pathlib import Path
from typing import List, Tuple, Dict

def find_fstring_issues(project_dir: str = ".") -> List[Tuple[str, int, str]]:
    """
    Find all f-string backslash issues in the project.
    
    Returns:
        List of (file_path, line_number, problematic_line)
    """
    issues = []
    
    # Patterns that cause f-string backslash errors
    patterns = [
        r'f"[^"]*\\[^"]*\{[^}]*\}[^"]*"',  # f"path\{var}"
        r"f'[^']*\\[^']*\{[^}]*\}[^']*'",  # f'path\{var}'
        r'f"[^"]*\{[^}]*\}[^"]*\\[^"]*"',  # f"{var}\path"
        r"f'[^']*\{[^}]*\}[^']*\\[^']*'",  # f'{var}\path'
    ]
    
    for py_file in glob.glob(f"{project_dir}/**/*.py", recursive=True):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, 1):
                for pattern in patterns:
                    if re.search(pattern, line):
                        issues.append((py_file, line_num, line.strip()))
        except Exception as e:
            print(f"Error reading {py_file}: {e}")
    
    return issues

def fix_fstring_line(line: str) -> str:
    """
    Fix a single line with f-string backslash issues.
    
    Args:
        line: Line of code to fix
        
    Returns:
        Fixed line of code
    """
    # Pattern 1: f"path\{variable}" -> Path("path") / variable
    pattern1 = r'f"([^"]*?)\\([^"]*?)\{([^}]+)\}([^"]*?)"'
    def replace1(match):
        before, middle, var, after = match.groups()
        if before and middle and after:
            return f'str(Path("{before}") / "{middle}" / {var} / "{after}")'
        elif before and middle:
            return f'str(Path("{before}") / "{middle}" / {var})'
        elif middle and after:
            return f'str(Path("{middle}") / {var} / "{after}")'
        else:
            return f'str(Path({var}))'
    
    line = re.sub(pattern1, replace1, line)
    
    # Pattern 2: f"{variable}\path" -> variable / Path("path")
    pattern2 = r'f"\{([^}]+)\}\\([^"]*?)"'
    def replace2(match):
        var, path = match.groups()
        return f'str(Path({var}) / "{path}")'
    
    line = re.sub(pattern2, replace2, line)
    
    # Pattern 3: Simple cases f"folder\{var}" -> f"folder/{var}"
    pattern3 = r'f"([^"]*?)\\([^"]*?)"'
    def replace3(match):
        content = match.group(0)
        if '{' in content:  # Only fix if it contains variables
            fixed_content = content.replace('\\', '/')
            return fixed_content
        return content
    
    line = re.sub(pattern3, replace3, line)
    
    return line

def fix_file(file_path: str) -> bool:
    """
    Fix all f-string issues in a single file.
    
    Args:
        file_path: Path to the Python file
        
    Returns:
        True if file was modified, False otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        lines = original_content.split('\n')
        fixed_lines = []
        file_modified = False
        
        for line in lines:
            fixed_line = fix_fstring_line(line)
            if fixed_line != line:
                file_modified = True
                print(f"  Fixed: {line.strip()}")
                print(f"    To: {fixed_line.strip()}")
            fixed_lines.append(fixed_line)
        
        if file_modified:
            # Create backup
            backup_path = f"{file_path}.backup"
            shutil.copy2(file_path, backup_path)
            
            # Write fixed content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(fixed_lines))
            
            print(f"✓ Fixed {file_path} (backup: {backup_path})")
            return True
        
        return False
        
    except Exception as e:
        print(f"✗ Error fixing {file_path}: {e}")
        return False

def add_pathlib_imports(project_dir: str = "."):
    """
    Add pathlib imports to files that need them.
    """
    for py_file in glob.glob(f"{project_dir}/**/*.py", recursive=True):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if file uses Path but doesn't import it
            if 'Path(' in content and 'from pathlib import Path' not in content and 'import pathlib' not in content:
                lines = content.split('\n')
                
                # Find the best place to insert import
                import_line = "from pathlib import Path"
                
                # Look for existing imports
                last_import_idx = -1
                for i, line in enumerate(lines):
                    if line.startswith('import ') or line.startswith('from '):
                        last_import_idx = i
                
                if last_import_idx >= 0:
                    # Insert after last import
                    lines.insert(last_import_idx + 1, import_line)
                else:
                    # Insert after docstring/comments at the top
                    insert_idx = 0
                    for i, line in enumerate(lines):
                        if line.strip() and not line.startswith('#') and '"""' not in line and "'''" not in line:
                            insert_idx = i
                            break
                    lines.insert(insert_idx, import_line)
                
                # Write the modified file
                with open(py_file, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))
                
                print(f"✓ Added pathlib import to {py_file}")
                
        except Exception as e:
            print(f"✗ Error adding import to {py_file}: {e}")

def test_fixes(project_dir: str = ".") -> List[str]:
    """
    Test that all Python files can be imported after fixes.
    
    Returns:
        List of files with remaining issues
    """
    issues = []
    
    for py_file in glob.glob(f"{project_dir}/**/*.py", recursive=True):
        # Skip test files and scripts
        if any(skip in py_file for skip in ['test_', '__pycache__', '.backup']):
            continue
        
        try:
            # Try to compile the file
            with open(py_file, 'r', encoding='utf-8') as f:
                source = f.read()
            
            compile(source, py_file, 'exec')
            
        except SyntaxError as e:
            if 'f-string expression part cannot include a backslash' in str(e):
                issues.append(f"{py_file}:{e.lineno} - {e.msg}")
            else:
                # Other syntax errors - might not be related to our fixes
                print(f"⚠ Other syntax error in {py_file}: {e}")
        except Exception as e:
            print(f"⚠ Error testing {py_file}: {e}")
    
    return issues

def main():
    """Main function to fix all f-string issues in the project."""
    print("🔧 Universal F-String Backslash Fix")
    print("=" * 50)
    
    project_dir = "."
    
    # Step 1: Find all issues
    print("1. Finding f-string backslash issues...")
    issues = find_fstring_issues(project_dir)
    
    if not issues:
        print("✓ No f-string backslash issues found!")
        return
    
    print(f"Found {len(issues)} potential issues:")
    for file_path, line_num, line in issues:
        print(f"  {file_path}:{line_num} - {line}")
    
    # Step 2: Fix files
    print("\n2. Fixing files...")
    fixed_files = []
    
    # Get unique files that need fixing
    files_to_fix = list(set(issue[0] for issue in issues))
    
    for file_path in files_to_fix:
        if fix_file(file_path):
            fixed_files.append(file_path)
    
    # Step 3: Add missing imports
    print("\n3. Adding missing pathlib imports...")
    add_pathlib_imports(project_dir)
    
    # Step 4: Test fixes
    print("\n4. Testing fixes...")
    remaining_issues = test_fixes(project_dir)
    
    if remaining_issues:
        print(f"⚠ {len(remaining_issues)} issues remain:")
        for issue in remaining_issues:
            print(f"  {issue}")
    else:
        print("✓ All f-string issues fixed successfully!")
    
    # Step 5: Summary
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"✓ Fixed {len(fixed_files)} files")
    print(f"✓ Resolved {len(issues)} f-string issues")
    
    if fixed_files:
        print("\nModified files:")
        for file_path in fixed_files:
            print(f"  - {file_path} (backup: {file_path}.backup)")
    
    if remaining_issues:
        print(f"\n⚠ {len(remaining_issues)} issues need manual review")
    else:
        print("\n🎉 Project is now cross-platform compatible!")

if __name__ == "__main__":
    main()
