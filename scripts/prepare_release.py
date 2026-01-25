"""
Release Preparation Script
自動的にバージョンアップとCHANGELOG更新をサポート
"""
import argparse
import re
import subprocess
from datetime import datetime
from pathlib import Path


def update_changelog(version: str, changes: str = None):
    """CHANGELOG.mdを更新"""
    changelog_path = Path("CHANGELOG.md")
    
    if not changelog_path.exists():
        print("Error: CHANGELOG.md not found")
        return False
    
    content = changelog_path.read_text(encoding='utf-8')
    
    # 既存のUnreleasedセクションを探す
    unreleased_pattern = r'## \[Unreleased\]\n\n'
    
    new_version_entry = f"""## [{version}] - {datetime.now().strftime('%Y-%m-%d')}

### Added
- 

### Changed
- 

### Fixed
- 

## [Unreleased]

"""
    
    # Unreleasedの後に新バージョンを挿入
    updated_content = re.sub(
        unreleased_pattern,
        new_version_entry,
        content,
        count=1
    )
    
    if updated_content == content:
        print("Warning: Could not find [Unreleased] section")
        return False
    
    changelog_path.write_text(updated_content, encoding='utf-8')
    print(f"✓ Updated CHANGELOG.md for version {version}")
    return True


def create_git_tag(version: str, message: str = None):
    """Gitタグを作成"""
    tag_name = f"v{version}"
    tag_message = message or f"Release version {version}"
    
    try:
        # タグ作成
        subprocess.run(
            ["git", "tag", "-a", tag_name, "-m", tag_message],
            check=True,
            capture_output=True
        )
        print(f"✓ Created git tag: {tag_name}")
        
        # リモートにプッシュ (オプション)
        push = input("Push tag to remote? (y/n): ")
        if push.lower() == 'y':
            subprocess.run(
                ["git", "push", "origin", tag_name],
                check=True
            )
            print(f"✓ Pushed tag {tag_name} to remote")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating tag: {e.stderr.decode()}")
        return False


def run_tests():
    """テストを実行"""
    print("Running tests...")
    try:
        result = subprocess.run(
            ["pytest", "core/tests", "-v"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✓ All tests passed")
            return True
        else:
            print("✗ Tests failed:")
            print(result.stdout)
            return False
    except FileNotFoundError:
        print("Warning: pytest not found, skipping tests")
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Prepare a new release version'
    )
    parser.add_argument(
        '--version',
        required=True,
        help='Version number (e.g., 0.5.0)'
    )
    parser.add_argument(
        '--skip-tests',
        action='store_true',
        help='Skip running tests'
    )
    parser.add_argument(
        '--message',
        help='Custom tag message'
    )
    
    args = parser.parse_args()
    
    print(f"\n=== Preparing release {args.version} ===\n")
    
    # Step 1: テスト実行
    if not args.skip_tests:
        if not run_tests():
            print("\nRelease preparation aborted due to test failures")
            return
    
    # Step 2: CHANGELOG更新
    if not update_changelog(args.version):
        print("\nPlease manually update CHANGELOG.md")
    
    # Step 3: ユーザーに編集を促す
    print("\n✎ Please edit CHANGELOG.md and add details for this release")
    input("Press Enter when ready to continue...")
    
    # Step 4: 変更をコミット
    try:
        subprocess.run(["git", "add", "CHANGELOG.md"], check=True)
        subprocess.run(
            ["git", "commit", "-m", f"chore: Prepare release {args.version}"],
            check=True
        )
        print("✓ Committed CHANGELOG changes")
    except subprocess.CalledProcessError:
        print("Warning: Could not commit changes (may already be committed)")
    
    # Step 5: タグ作成
    create_git_tag(args.version, args.message)
    
    print(f"\n✓ Release {args.version} preparation complete!")
    print("\nNext steps:")
    print("1. Review the changes")
    print("2. Push to remote: git push origin main")
    print(f"3. Create GitHub release for tag v{args.version}")


if __name__ == "__main__":
    main()
