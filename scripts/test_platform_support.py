"""Regression checks for study-specific platform requirements; no experiments."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from platform_support import check_platform


class PlatformRequirements(unittest.TestCase):
    def check(self, study=None, bits=53, fork=False):
        numpy = SimpleNamespace(longdouble=object(),
                                finfo=lambda dtype: SimpleNamespace(nmant=bits - 1))
        methods = ['spawn', 'fork'] if fork else ['spawn']
        with patch.dict(sys.modules, {'numpy': numpy}), \
             patch('platform_support.multiprocessing.get_all_start_methods', return_value=methods):
            return check_platform() if study is None else check_platform(study)

    def test_portable_studies_accept_binary64_without_fork(self):
        for study in ('continuation', 'tubes'):
            with self.subTest(study=study):
                result = self.check(study)
                self.assertEqual(result['study'], study)
                self.assertEqual(result['longdouble_significand_bits'], 53)
                self.assertFalse(result['fork_available'])
                self.assertFalse(result['requires_extended_precision'])
                self.assertFalse(result['requires_fork'])

    def test_default_full_suite_stays_restricted(self):
        with self.assertRaisesRegex(RuntimeError, '53-bit significand'):
            self.check()

    def test_other_studies_stay_restricted(self):
        for study in ('all', 'baseline', 'schwarz', 'accurate', 'flux', 'controls',
                      'radial', 'diagnostics', 'conditioning'):
            with self.subTest(study=study):
                with self.assertRaisesRegex(RuntimeError, '53-bit significand'):
                    self.check(study, fork=True)

    def test_full_suite_still_requires_fork(self):
        with self.assertRaisesRegex(RuntimeError, 'fork is unavailable'):
            self.check(bits=64)

    def test_supported_full_suite(self):
        result = self.check(bits=64, fork=True)
        self.assertEqual(result['study'], 'all')
        self.assertTrue(result['requires_extended_precision'])
        self.assertTrue(result['requires_fork'])

    def test_other_arithmetic_is_not_implicitly_validated(self):
        for bits in (24, 113):
            with self.subTest(bits=bits):
                with self.assertRaises(RuntimeError):
                    self.check('baseline', bits=bits, fork=True)

    def test_unknown_study_does_not_skip_requirements(self):
        with self.assertRaisesRegex(ValueError, 'Unknown study'):
            self.check('unknown-study')


if __name__ == '__main__':
    unittest.main()
