import unittest

from experiment.config import read_config


class TestConfig(unittest.TestCase):
    def test_read_config_simple(self):
        config_str = """
            Projects:
              C/C++ Libraries:
              - libyaml       # "C" Fast YAML 1.1
              - Syck          # (dated) "C" YAML 1.0
              - yaml-cpp      # C++ YAML 1.2 implementation
        """
        config = read_config(config_str)
        self.assertDictEqual(config, {
            'Projects': {
                'C/C++ Libraries': ['libyaml', 'Syck', 'yaml-cpp']
            }
        })

    def test_read_config_merge(self):
        config_default_str = """
            Projects:
              C/C++ Libraries:
              - libyaml       # "C" Fast YAML 1.1
              - Syck          # (dated) "C" YAML 1.0
              - yaml-cpp      # C++ YAML 1.2 implementation
              Other Libraries:
              - pyyaml
        """
        config_str = """
            Projects:
              C/C++ Libraries:
            Something: Test
        """
        config = read_config(config_str, config_default_str)
        self.assertDictEqual(config, {
            'Projects': {
                'C/C++ Libraries': None,
                'Other Libraries': ['pyyaml']
            },
            'Something': 'Test'
        })
