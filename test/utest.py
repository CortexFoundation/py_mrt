import unittest

if __name__ == '__main__':
    suite = unittest.TestSuite()

    suite.addTests(unittest.TestLoader().discover('./fquant', 't_*.py', top_level_dir=None))

    runner = unittest.TextTestRunner(verbosity=1)
    runner.run(suite)
