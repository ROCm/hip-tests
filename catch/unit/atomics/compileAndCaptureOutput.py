# Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import os
import subprocess
import sys
import unittest

class AtomicsCompile(unittest.TestCase):
  path = None
  expected_error_count = None
  file = None
  error_string = None

  def setUp(self):
    self.error_string = 'error:'
    self.assertFalse(self.path == None)
    self.assertFalse(self.file == None)
    self.assertFalse(self.expected_error_count == None)

  def test_atomic(self):
    compiler_output = subprocess.run([os.environ["HIP_PATH"] + '/bin/hipcc',
      '-I' + self.path + '/../../external/Catch2',
      '-I' + self.path + '/../../include',
      '-I' + os.environ["HIP_PATH"] + '/include',
      '-I' + self.path + '/../../external/picojson',
      '--std=c++17',
      '-c',
      self.path + '/' + self.file],
      stderr=subprocess.PIPE)
    # Get the compiler output in the stdout if -V flag is raised during ctest invocation.
    print(compiler_output.stderr.decode('UTF-8'))
    self.assertEqual(compiler_output.stderr.decode('UTF-8').count(self.error_string),
                     self.expected_error_count)

if __name__ == '__main__':
  if len(sys.argv) == 4:
    AtomicsCompile.path = sys.argv[1]
    AtomicsCompile.file = sys.argv[2]
    AtomicsCompile.expected_error_count = int(sys.argv[3])
  # Unittest looks at the same argv's as the __main__ and doesn't know how
  # to handle arguments other than the executable (0). Therefore passing only
  # executable as the argv for unittest module.
  unittest.main(argv=[sys.argv[0]])
