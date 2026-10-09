import unittest

from pathlib import Path
from tempfile import TemporaryDirectory

from app.repository import build_repository_paths, parse_github_repo_url, resolve_repo_file


class RepositoryParsingTests(unittest.TestCase):
    def test_parse_github_url_normalizes_to_git_url(self):
        repo = parse_github_repo_url("https://github.com/humeyrapolat/codecontext")

        self.assertEqual(repo.owner, "humeyrapolat")
        self.assertEqual(repo.name, "codecontext")
        self.assertEqual(repo.normalized_url, "https://github.com/humeyrapolat/codecontext.git")
        self.assertTrue(repo.repo_id.startswith("humeyrapolat-codecontext-"))

    def test_parse_github_url_rejects_non_github_hosts(self):
        with self.assertRaises(ValueError):
            parse_github_repo_url("https://example.com/humeyrapolat/codecontext")

    def test_parse_github_url_rejects_incomplete_paths(self):
        with self.assertRaises(ValueError):
            parse_github_repo_url("https://github.com/humeyrapolat")

    def test_repository_paths_are_repo_scoped(self):
        paths = build_repository_paths("repo-123", data_dir="tmp-data")

        self.assertEqual(str(paths.source_dir), "tmp-data/repositories/repo-123/source")
        self.assertEqual(str(paths.index_dir), "tmp-data/repositories/repo-123/faiss")

    def test_resolve_repo_file_allows_files_inside_repo(self):
        with TemporaryDirectory() as tmp_dir:
            repo_dir = Path(tmp_dir) / "repo"
            repo_dir.mkdir()
            target = repo_dir / "app.py"
            target.write_text("print('ok')", encoding="utf-8")

            resolved = resolve_repo_file(repo_dir, "app.py")

            self.assertEqual(resolved, target.resolve())

    def test_resolve_repo_file_rejects_parent_traversal(self):
        with TemporaryDirectory() as tmp_dir:
            repo_dir = Path(tmp_dir) / "repo"
            repo_dir.mkdir()

            with self.assertRaises(ValueError):
                resolve_repo_file(repo_dir, "../secret.txt")


if __name__ == "__main__":
    unittest.main()
