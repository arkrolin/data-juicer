import os
import tempfile
import unittest

from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class ParseRequirementsFileTest(DataJuicerTestCaseBase):
    """Test parse_requirements_file with real temp files."""

    def setUp(self):
        super().setUp()
        from data_juicer.ops.op_env import parse_requirements_file
        self.parse_requirements_file = parse_requirements_file
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        super().tearDown()
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_basic_requirements_file(self):
        req_file = os.path.join(self.tmp_dir, "requirements.txt")
        with open(req_file, "w") as f:
            f.write("numpy>=1.20.0\npandas>=1.3.0\nscipy\n")
        result = self.parse_requirements_file(req_file)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].name, "numpy")
        self.assertEqual(result[1].name, "pandas")
        self.assertEqual(result[2].name, "scipy")

    def test_requirements_file_with_comments_and_blanks(self):
        req_file = os.path.join(self.tmp_dir, "requirements.txt")
        with open(req_file, "w") as f:
            f.write("# This is a comment\n\nnumpy>=1.20.0\n\n# Another comment\npandas\n")
        result = self.parse_requirements_file(req_file)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].name, "numpy")
        self.assertEqual(result[1].name, "pandas")

    def test_requirements_file_with_extras(self):
        req_file = os.path.join(self.tmp_dir, "requirements.txt")
        with open(req_file, "w") as f:
            f.write("requests[security]>=2.25.0\nflask[async]\n")
        result = self.parse_requirements_file(req_file)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].name, "requests")
        self.assertEqual(result[0].extras, ["security"])
        self.assertEqual(result[1].name, "flask")
        self.assertEqual(result[1].extras, ["async"])

    def test_requirements_file_with_git_url(self):
        req_file = os.path.join(self.tmp_dir, "requirements.txt")
        with open(req_file, "w") as f:
            f.write("git+https://github.com/user/repo.git\nnumpy\n")
        result = self.parse_requirements_file(req_file)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].url, "git+https://github.com/user/repo.git")
        self.assertEqual(result[1].name, "numpy")

    def test_requirements_file_with_editable_local(self):
        local_pkg = os.path.join(self.tmp_dir, "local_pkg")
        os.makedirs(local_pkg)
        req_file = os.path.join(self.tmp_dir, "requirements.txt")
        with open(req_file, "w") as f:
            f.write(f"-e {local_pkg}\nnumpy\n")
        result = self.parse_requirements_file(req_file)
        self.assertEqual(len(result), 2)
        self.assertTrue(result[0].is_editable)
        self.assertTrue(result[0].is_local)
        self.assertEqual(result[0].path, local_pkg)
        self.assertEqual(result[1].name, "numpy")

    def test_empty_requirements_file(self):
        req_file = os.path.join(self.tmp_dir, "requirements.txt")
        with open(req_file, "w") as f:
            f.write("# only comments\n\n# nothing else\n")
        result = self.parse_requirements_file(req_file)
        self.assertEqual(len(result), 0)

    def test_requirements_file_with_version_specifiers(self):
        req_file = os.path.join(self.tmp_dir, "requirements.txt")
        with open(req_file, "w") as f:
            f.write("numpy>=1.20.0,<2.0\npandas==1.5.0\nscipy!=1.7.0\n")
        result = self.parse_requirements_file(req_file)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].name, "numpy")
        self.assertIn(">=1.20.0", str(result[0].version))
        self.assertEqual(result[1].name, "pandas")
        self.assertIn("==1.5.0", str(result[1].version))
        self.assertEqual(result[2].name, "scipy")
        self.assertIn("!=1.7.0", str(result[2].version))


class RequirementPostInitTest(DataJuicerTestCaseBase):
    """Test Requirement dataclass edge cases."""

    def setUp(self):
        super().setUp()
        from data_juicer.ops.op_env import Requirement
        self.Requirement = Requirement

    def test_version_string_converted_to_specifier_set(self):
        from packaging.specifiers import SpecifierSet
        req = self.Requirement(name="numpy", version=">=1.20.0")
        self.assertIsInstance(req.version, SpecifierSet)

    def test_version_none_remains_none(self):
        req = self.Requirement(name="numpy")
        self.assertIsNone(req.version)

    def test_requirement_str_no_name_no_url(self):
        req = self.Requirement()
        self.assertEqual(str(req), "")

    def test_requirement_str_name_only(self):
        req = self.Requirement(name="numpy")
        self.assertEqual(str(req), "numpy")

    def test_requirement_str_with_multiple_extras(self):
        req = self.Requirement(name="pkg", version=">=1.0", extras=["extra1", "extra2"])
        result = str(req)
        self.assertIn("pkg[", result)
        self.assertIn("extra1", result)
        self.assertIn("extra2", result)
        self.assertIn(">=1.0", result)

    def test_requirement_str_url_with_name(self):
        req = self.Requirement(name="mypkg", url="https://example.com/pkg.tar.gz")
        self.assertEqual(str(req), "mypkg @ https://example.com/pkg.tar.gz")

    def test_requirement_str_url_without_name(self):
        req = self.Requirement(url="https://example.com/pkg.tar.gz")
        self.assertEqual(str(req), "https://example.com/pkg.tar.gz")


class ParseSingleRequirementEdgeCasesTest(DataJuicerTestCaseBase):
    """Test parse_single_requirement edge cases not covered by existing tests."""

    def setUp(self):
        super().setUp()
        from data_juicer.ops.op_env import parse_single_requirement
        self.parse_single_requirement = parse_single_requirement

    def test_parse_invalid_requirement_returns_none(self):
        result = self.parse_single_requirement("!!!invalid!!!")
        self.assertIsNone(result)

    def test_parse_requirement_with_marker(self):
        req = self.parse_single_requirement("numpy>=1.20.0; python_version>='3.8'")
        self.assertIsNotNone(req)
        self.assertEqual(req.name, "numpy")
        self.assertIsNotNone(req.markers)

    def test_parse_requirement_with_multiple_version_specs(self):
        req = self.parse_single_requirement("numpy>=1.20.0,<2.0,!=1.24.0")
        self.assertIsNotNone(req)
        self.assertEqual(req.name, "numpy")
        self.assertIn(">=1.20.0", str(req.version))
        self.assertIn("<2.0", str(req.version))
        self.assertIn("!=1.24.0", str(req.version))

    def test_parse_requirement_with_url_specifier(self):
        req = self.parse_single_requirement("mypkg @ https://example.com/pkg-1.0.tar.gz")
        self.assertIsNotNone(req)
        self.assertEqual(req.name, "mypkg")
        self.assertEqual(req.url, "https://example.com/pkg-1.0.tar.gz")

    def test_parse_requirement_git_at(self):
        req = self.parse_single_requirement("git@github.com:user/repo.git")
        self.assertIsNotNone(req)
        self.assertEqual(req.url, "git@github.com:user/repo.git")

    def test_parse_local_directory(self):
        tmp_dir = tempfile.mkdtemp()
        try:
            req = self.parse_single_requirement(tmp_dir)
            self.assertIsNotNone(req)
            self.assertTrue(req.is_local)
            self.assertEqual(req.path, tmp_dir)
            self.assertFalse(req.is_editable)
        finally:
            os.rmdir(tmp_dir)

    def test_parse_whitespace_handling(self):
        req = self.parse_single_requirement("  numpy >= 1.20.0  ")
        self.assertIsNotNone(req)
        self.assertEqual(req.name, "numpy")


class OPEnvSpecEdgeCasesTest(DataJuicerTestCaseBase):
    """Test OPEnvSpec edge cases."""

    def setUp(self):
        super().setUp()
        from data_juicer.ops.op_env import OPEnvSpec, Requirement
        self.OPEnvSpec = OPEnvSpec
        self.Requirement = Requirement

    def test_get_hash_changes_with_env_vars(self):
        spec1 = self.OPEnvSpec(pip_pkgs=["numpy>=1.20.0"])
        spec2 = self.OPEnvSpec(pip_pkgs=["numpy>=1.20.0"], env_vars={"KEY": "val"})
        self.assertNotEqual(spec1.get_hash(), spec2.get_hash())

    def test_get_hash_changes_with_working_dir(self):
        spec1 = self.OPEnvSpec(pip_pkgs=["numpy>=1.20.0"])
        spec2 = self.OPEnvSpec(pip_pkgs=["numpy>=1.20.0"], working_dir="/some/path")
        self.assertNotEqual(spec1.get_hash(), spec2.get_hash())

    def test_to_dict_empty(self):
        spec = self.OPEnvSpec()
        self.assertEqual(spec.to_dict(), {})

    def test_to_dict_with_extra_env_params(self):
        spec = self.OPEnvSpec(
            pip_pkgs=["numpy"],
            extra_env_params={"excludes": ["bad_pkg"]}
        )
        d = spec.to_dict()
        self.assertIn("uv", d)
        self.assertIn("excludes", d)
        self.assertEqual(d["excludes"], ["bad_pkg"])

    def test_parsed_requirements_from_list(self):
        spec = self.OPEnvSpec(pip_pkgs=["numpy>=1.20.0", "pandas==1.5.0"])
        self.assertIn("numpy", spec.parsed_requirements)
        self.assertIn("pandas", spec.parsed_requirements)
        self.assertEqual(spec.parsed_requirements["numpy"].name, "numpy")

    def test_init_with_none_pip_pkgs(self):
        spec = self.OPEnvSpec(pip_pkgs=None)
        self.assertEqual(spec.pip_pkgs, [])

    def test_backend_pip(self):
        spec = self.OPEnvSpec(pip_pkgs=["numpy"], backend="pip")
        d = spec.to_dict()
        self.assertIn("pip", d)
        self.assertNotIn("uv", d)


class OPEnvManagerCombineEdgeCasesTest(DataJuicerTestCaseBase):
    """Test OPEnvManager combination and merge edge cases."""

    def setUp(self):
        super().setUp()
        from data_juicer.ops.op_env import OPEnvManager, OPEnvSpec, ConflictResolveStrategy
        self.OPEnvManager = OPEnvManager
        self.OPEnvSpec = OPEnvSpec
        self.ConflictResolveStrategy = ConflictResolveStrategy

    def test_init_with_string_strategy(self):
        manager = self.OPEnvManager(conflict_resolve_strategy="split")
        self.assertEqual(manager.conflict_resolve_strategy, self.ConflictResolveStrategy.SPLIT)

    def test_init_with_string_strategy_latest(self):
        manager = self.OPEnvManager(conflict_resolve_strategy="latest")
        self.assertEqual(manager.conflict_resolve_strategy, self.ConflictResolveStrategy.LATEST)

    def test_init_with_string_strategy_overwrite(self):
        manager = self.OPEnvManager(conflict_resolve_strategy="overwrite")
        self.assertEqual(manager.conflict_resolve_strategy, self.ConflictResolveStrategy.OVERWRITE)

    def test_try_combine_different_backends(self):
        manager = self.OPEnvManager(min_common_dep_num_to_combine=1)
        spec1 = self.OPEnvSpec(pip_pkgs=["numpy>=1.20.0"], backend="pip")
        spec2 = self.OPEnvSpec(pip_pkgs=["numpy>=1.19.0"], backend="uv")

        manager.record_op_env_spec("op1", spec1)
        manager.record_op_env_spec("op2", spec2)

        # When backends differ, should combine using "uv" as default
        combined_spec = manager.hash2specs[manager.op2hash["op1"]]
        self.assertEqual(combined_spec.backend, "uv")

    def test_try_combine_second_has_env_vars_only(self):
        manager = self.OPEnvManager(min_common_dep_num_to_combine=1)
        spec1 = self.OPEnvSpec(pip_pkgs=["numpy>=1.20.0"])
        spec2 = self.OPEnvSpec(pip_pkgs=["numpy>=1.19.0"], env_vars={"KEY": "val"})

        manager.record_op_env_spec("op1", spec1)
        manager.record_op_env_spec("op2", spec2)

        combined_spec = manager.hash2specs[manager.op2hash["op1"]]
        self.assertEqual(combined_spec.env_vars, {"KEY": "val"})

    def test_try_combine_working_dir_from_first(self):
        manager = self.OPEnvManager(min_common_dep_num_to_combine=1)
        spec1 = self.OPEnvSpec(pip_pkgs=["numpy>=1.20.0"], working_dir="/tmp/work")
        spec2 = self.OPEnvSpec(pip_pkgs=["numpy>=1.19.0"])

        manager.record_op_env_spec("op1", spec1)
        manager.record_op_env_spec("op2", spec2)

        combined_spec = manager.hash2specs[manager.op2hash["op1"]]
        self.assertEqual(combined_spec.working_dir, "/tmp/work")

    def test_try_combine_working_dir_from_second(self):
        manager = self.OPEnvManager(min_common_dep_num_to_combine=1)
        spec1 = self.OPEnvSpec(pip_pkgs=["numpy>=1.20.0"])
        spec2 = self.OPEnvSpec(pip_pkgs=["numpy>=1.19.0"], working_dir="/tmp/work2")

        manager.record_op_env_spec("op1", spec1)
        manager.record_op_env_spec("op2", spec2)

        combined_spec = manager.hash2specs[manager.op2hash["op1"]]
        self.assertEqual(combined_spec.working_dir, "/tmp/work2")

    def test_record_same_hash_spec_twice(self):
        """Recording a spec that already has the same hash should not duplicate."""
        manager = self.OPEnvManager(min_common_dep_num_to_combine=-1)
        spec1 = self.OPEnvSpec(pip_pkgs=["numpy>=1.20.0"])
        spec2 = self.OPEnvSpec(pip_pkgs=["numpy>=1.20.0"])  # Same content

        manager.record_op_env_spec("op1", spec1)
        manager.record_op_env_spec("op2", spec2)

        self.assertEqual(manager.op2hash["op1"], manager.op2hash["op2"])
        self.assertEqual(len(manager.hash2specs), 1)

    def test_merge_three_ops_incrementally(self):
        """Three ops with common deps combine incrementally."""
        manager = self.OPEnvManager(min_common_dep_num_to_combine=1)
        spec1 = self.OPEnvSpec(pip_pkgs=["numpy>=1.20.0", "requests"])
        spec2 = self.OPEnvSpec(pip_pkgs=["numpy>=1.19.0", "flask"])
        spec3 = self.OPEnvSpec(pip_pkgs=["numpy>=1.18.0", "click"])

        manager.record_op_env_spec("op1", spec1)
        manager.record_op_env_spec("op2", spec2)
        manager.record_op_env_spec("op3", spec3)

        # All three should share the same hash
        self.assertEqual(manager.op2hash["op1"], manager.op2hash["op2"])
        self.assertEqual(manager.op2hash["op2"], manager.op2hash["op3"])
        # Only one active hash group remains
        self.assertEqual(len(manager.hash2ops), 1)

        combined_spec = manager.hash2specs[manager.op2hash["op1"]]
        req_names = combined_spec.get_requirement_name_list()
        self.assertIn("numpy", req_names)
        self.assertIn("requests", req_names)
        self.assertIn("flask", req_names)
        self.assertIn("click", req_names)

    def test_no_common_deps_not_combined(self):
        """Specs with no common deps stay separate even with min_common=0."""
        manager = self.OPEnvManager(min_common_dep_num_to_combine=1)
        spec1 = self.OPEnvSpec(pip_pkgs=["numpy>=1.20.0"])
        spec2 = self.OPEnvSpec(pip_pkgs=["pandas>=1.3.0"])

        manager.record_op_env_spec("op1", spec1)
        manager.record_op_env_spec("op2", spec2)

        self.assertNotEqual(manager.op2hash["op1"], manager.op2hash["op2"])
        self.assertEqual(len(manager.hash2specs), 2)

    def test_min_common_dep_zero_combines_all(self):
        """With min_common_dep_num_to_combine=0, all specs combine."""
        manager = self.OPEnvManager(min_common_dep_num_to_combine=0)
        spec1 = self.OPEnvSpec(pip_pkgs=["numpy>=1.20.0"])
        spec2 = self.OPEnvSpec(pip_pkgs=["pandas>=1.3.0"])

        manager.record_op_env_spec("op1", spec1)
        manager.record_op_env_spec("op2", spec2)

        self.assertEqual(manager.op2hash["op1"], manager.op2hash["op2"])
        self.assertEqual(len(manager.hash2ops), 1)


class ConflictResolveStrategyEnumTest(DataJuicerTestCaseBase):
    """Test ConflictResolveStrategy enum."""

    def setUp(self):
        super().setUp()
        from data_juicer.ops.op_env import ConflictResolveStrategy
        self.ConflictResolveStrategy = ConflictResolveStrategy

    def test_enum_values(self):
        self.assertEqual(self.ConflictResolveStrategy.SPLIT.value, "split")
        self.assertEqual(self.ConflictResolveStrategy.OVERWRITE.value, "overwrite")
        self.assertEqual(self.ConflictResolveStrategy.LATEST.value, "latest")

    def test_enum_from_value(self):
        self.assertEqual(self.ConflictResolveStrategy("split"), self.ConflictResolveStrategy.SPLIT)
        self.assertEqual(self.ConflictResolveStrategy("overwrite"), self.ConflictResolveStrategy.OVERWRITE)
        self.assertEqual(self.ConflictResolveStrategy("latest"), self.ConflictResolveStrategy.LATEST)

    def test_enum_invalid_value(self):
        with self.assertRaises(ValueError):
            self.ConflictResolveStrategy("invalid")


class ResolveWithStrategySupplementalTest(DataJuicerTestCaseBase):
    """Test _resolve_with_strategy with supplemental edge cases."""

    def setUp(self):
        super().setUp()
        from data_juicer.ops.op_env import OPEnvManager, Requirement, ConflictResolveStrategy
        self.OPEnvManager = OPEnvManager
        self.Requirement = Requirement
        self.ConflictResolveStrategy = ConflictResolveStrategy

    def test_resolve_both_none_versions(self):
        manager = self.OPEnvManager()
        req1 = self.Requirement(name="numpy")
        req2 = self.Requirement(name="numpy")
        result = manager._resolve_with_strategy(req1, req2)
        # version1 is None, returns second_req
        self.assertEqual(result, req2)

    def test_resolve_first_none_version(self):
        manager = self.OPEnvManager()
        req1 = self.Requirement(name="numpy")
        req2 = self.Requirement(name="numpy", version=">=1.20.0")
        result = manager._resolve_with_strategy(req1, req2)
        self.assertEqual(result, req2)

    def test_resolve_second_none_version(self):
        manager = self.OPEnvManager()
        req1 = self.Requirement(name="numpy", version=">=1.20.0")
        req2 = self.Requirement(name="numpy")
        result = manager._resolve_with_strategy(req1, req2)
        self.assertEqual(result, req1)

    def test_resolve_compatible_versions_merge(self):
        manager = self.OPEnvManager()
        req1 = self.Requirement(name="numpy", version=">=1.20.0")
        req2 = self.Requirement(name="numpy", version="<2.0")
        result = manager._resolve_with_strategy(req1, req2)
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "numpy")
        self.assertIn(">=1.20.0", str(result.version))
        self.assertIn("<2.0", str(result.version))

    def test_resolve_overwrite_with_conflict(self):
        """OVERWRITE returns second_req directly for conflicting versions."""
        manager = self.OPEnvManager(conflict_resolve_strategy="overwrite")
        req1 = self.Requirement(name="pkg", version=">=1.0,<2.0", extras=["a"])
        req2 = self.Requirement(name="pkg", version=">=3.0", extras=["b"])
        result = manager._resolve_with_strategy(req1, req2)
        # OVERWRITE returns second_req directly
        self.assertEqual(result, req2)
        self.assertEqual(result.extras, ["b"])

    def test_resolve_overwrite_compatible_combines_extras(self):
        """Compatible versions combine extras in the merged result."""
        manager = self.OPEnvManager(conflict_resolve_strategy="overwrite")
        req1 = self.Requirement(name="pkg", version=">=1.0", extras=["a"])
        req2 = self.Requirement(name="pkg", version="<2.0", extras=["b"])
        result = manager._resolve_with_strategy(req1, req2)
        # Compatible versions => merge (not OVERWRITE path)
        self.assertIsNotNone(result)
        self.assertIn("a", result.extras)
        self.assertIn("b", result.extras)

    def test_resolve_latest_with_single_range(self):
        """LATEST strategy with single unified range (not UnionSpecifier)."""
        manager = self.OPEnvManager(conflict_resolve_strategy="latest")
        req1 = self.Requirement(name="numpy", version=">=2.0")
        req2 = self.Requirement(name="numpy", version="<1.0")
        result = manager._resolve_with_strategy(req1, req2)
        # Conflict: >=2.0 and <1.0 are incompatible.
        # Union should produce a UnionSpecifier with unbounded range >=2.0
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "numpy")

    def test_resolve_url_preserved(self):
        manager = self.OPEnvManager()
        req1 = self.Requirement(name="pkg", version=">=1.0", url="https://example.com/pkg.tar.gz")
        req2 = self.Requirement(name="pkg", version=">=1.0,<2.0")
        result = manager._resolve_with_strategy(req1, req2)
        self.assertIsNotNone(result)
        self.assertEqual(result.url, "https://example.com/pkg.tar.gz")

    def test_resolve_editable_preserved(self):
        manager = self.OPEnvManager()
        req1 = self.Requirement(name="pkg", version=">=1.0", is_editable=True)
        req2 = self.Requirement(name="pkg", version=">=1.0,<2.0")
        result = manager._resolve_with_strategy(req1, req2)
        self.assertIsNotNone(result)
        self.assertTrue(result.is_editable)


class AnalyzeLazyLoadedRequirementsSupplementalTest(DataJuicerTestCaseBase):
    """Test analyze_lazy_loaded_requirements_for_code_file with real temp files."""

    def setUp(self):
        super().setUp()
        from data_juicer.ops.op_env import (
            analyze_lazy_loaded_requirements,
            analyze_lazy_loaded_requirements_for_code_file,
        )
        self.analyze_lazy_loaded_requirements = analyze_lazy_loaded_requirements
        self.analyze_lazy_loaded_requirements_for_code_file = analyze_lazy_loaded_requirements_for_code_file
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        super().tearDown()
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_file_with_only_module_name_no_package_name(self):
        """When only module_name is given, it should be used as the requirement name."""
        code_file = os.path.join(self.tmp_dir, "test_code.py")
        with open(code_file, "w") as f:
            f.write("""
from data_juicer.utils.lazy_loader import LazyLoader

cv2 = LazyLoader('cv2')
""")
        result = self.analyze_lazy_loaded_requirements_for_code_file(code_file)
        self.assertEqual(result, ["cv2"])

    def test_file_with_module_and_package_name_positional(self):
        """Positional args: module_name, package_name."""
        code_file = os.path.join(self.tmp_dir, "test_code.py")
        with open(code_file, "w") as f:
            f.write("""
from data_juicer.utils.lazy_loader import LazyLoader

cv2 = LazyLoader('cv2', 'opencv-python')
""")
        result = self.analyze_lazy_loaded_requirements_for_code_file(code_file)
        self.assertEqual(result, ["opencv-python"])

    def test_file_with_package_url_kwarg(self):
        code_file = os.path.join(self.tmp_dir, "test_code.py")
        with open(code_file, "w") as f:
            f.write("""
from data_juicer.utils.lazy_loader import LazyLoader

custom = LazyLoader('custom_mod', package_url='https://github.com/user/custom.git')
""")
        result = self.analyze_lazy_loaded_requirements_for_code_file(code_file)
        self.assertEqual(result, ["custom_mod @ https://github.com/user/custom.git"])

    def test_file_with_check_packages_kwarg(self):
        code_file = os.path.join(self.tmp_dir, "test_code.py")
        with open(code_file, "w") as f:
            f.write("""
from data_juicer.utils.lazy_loader import LazyLoader

LazyLoader.check_packages(package_specs=['torch>=2.0', 'torchvision'])
""")
        result = self.analyze_lazy_loaded_requirements_for_code_file(code_file)
        self.assertEqual(sorted(result), sorted(["torch>=2.0", "torchvision"]))

    def test_file_with_no_lazy_loader_imports(self):
        code_file = os.path.join(self.tmp_dir, "test_code.py")
        with open(code_file, "w") as f:
            f.write("""
import numpy as np
import pandas as pd

def compute():
    return np.array([1, 2, 3])
""")
        result = self.analyze_lazy_loaded_requirements_for_code_file(code_file)
        self.assertEqual(result, [])

    def test_file_with_multiple_check_packages_calls(self):
        code_file = os.path.join(self.tmp_dir, "test_code.py")
        with open(code_file, "w") as f:
            f.write("""
from data_juicer.utils.lazy_loader import LazyLoader

LazyLoader.check_packages(['numpy>=1.20.0'])
LazyLoader.check_packages(['pandas>=1.3.0'])
LazyLoader.check_packages(package_specs=['scipy'])
""")
        result = self.analyze_lazy_loaded_requirements_for_code_file(code_file)
        self.assertEqual(sorted(result), sorted(["numpy>=1.20.0", "pandas>=1.3.0", "scipy"]))

    def test_mixed_lazy_loader_and_check_packages_in_file(self):
        code_file = os.path.join(self.tmp_dir, "test_code.py")
        with open(code_file, "w") as f:
            f.write("""
from data_juicer.utils.lazy_loader import LazyLoader

transformers = LazyLoader('transformers', 'transformers')
LazyLoader.check_packages(['tokenizers>=0.13.0'])
torch = LazyLoader('torch', package_name='torch')
""")
        result = self.analyze_lazy_loaded_requirements_for_code_file(code_file)
        self.assertEqual(sorted(result), sorted(["transformers", "tokenizers>=0.13.0", "torch"]))

    def test_code_with_other_function_calls_ignored(self):
        """Other function calls should not be picked up."""
        code_content = """
from data_juicer.utils.lazy_loader import LazyLoader

numpy = LazyLoader('numpy', 'numpy')
result = some_other_function('arg1', 'arg2')
another_call(package_name='irrelevant')
"""
        result = self.analyze_lazy_loaded_requirements(code_content)
        self.assertEqual(result, ["numpy"])

    def test_code_with_all_three_positional_args(self):
        """Three positional args: module_name, package_name, package_url."""
        code_content = """
from data_juicer.utils.lazy_loader import LazyLoader

pkg = LazyLoader('mod_name', 'pkg_name', 'https://example.com/repo.git')
"""
        result = self.analyze_lazy_loaded_requirements(code_content)
        self.assertEqual(result, ["pkg_name @ https://example.com/repo.git"])


class OPEnvManagerPrintStatesTest(DataJuicerTestCaseBase):
    """Test OPEnvManager.print_the_current_states output."""

    def setUp(self):
        super().setUp()
        from data_juicer.ops.op_env import OPEnvManager, OPEnvSpec
        self.OPEnvManager = OPEnvManager
        self.OPEnvSpec = OPEnvSpec

    def test_print_states_empty(self):
        manager = self.OPEnvManager()
        states = manager.print_the_current_states()
        self.assertEqual(len(states), 0)

    def test_print_states_single_op(self):
        manager = self.OPEnvManager()
        spec = self.OPEnvSpec(pip_pkgs=["numpy>=1.20.0"])
        manager.record_op_env_spec("my_op", spec)
        states = manager.print_the_current_states()
        self.assertEqual(len(states), 1)
        # The key should be the op name
        self.assertIn("my_op", list(states.keys())[0])

    def test_print_states_multiple_ops_same_spec(self):
        manager = self.OPEnvManager()
        spec = self.OPEnvSpec(pip_pkgs=["numpy>=1.20.0"])
        manager.record_op_env_spec("op_a", spec)
        manager.record_op_env_spec("op_b", spec)
        states = manager.print_the_current_states()
        self.assertEqual(len(states), 1)
        key = list(states.keys())[0]
        self.assertIn("op_a", key)
        self.assertIn("op_b", key)


class ParseRequirementsListEdgeCasesTest(DataJuicerTestCaseBase):
    """Test parse_requirements_list edge cases."""

    def setUp(self):
        super().setUp()
        from data_juicer.ops.op_env import parse_requirements_list
        self.parse_requirements_list = parse_requirements_list

    def test_empty_list(self):
        result = self.parse_requirements_list([])
        self.assertEqual(result, [])

    def test_list_with_invalid_entries(self):
        result = self.parse_requirements_list(["numpy>=1.20.0", "!!!invalid!!!", "pandas"])
        # Should parse valid ones and skip invalid
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].name, "numpy")
        self.assertEqual(result[1].name, "pandas")

    def test_list_with_all_invalid(self):
        result = self.parse_requirements_list(["!!!a!!!", "!!!b!!!"])
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
