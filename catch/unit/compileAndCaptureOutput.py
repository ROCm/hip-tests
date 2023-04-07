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

class CompileAndCapture(unittest.TestCase):
  path = None
  expected_error_count = None
  file = None
  error_string = None
  platform = None

  def setUp(self):
    self.error_string = 'error:'
    self.assertFalse(self.path == None)
    self.assertFalse(self.file == None)
    self.assertFalse(self.expected_error_count == None)
    self.assertTrue(self.platform == 'amd' or self.platform == 'nvidia')

  def test_atomic(self):
    compiler_args = [
      os.environ["HIP_PATH"] + '/bin/hipcc',
      '-I' + self.path + '/../../external/Catch2',
      '-I' + self.path + '/../../include',
      '-I' + os.environ["HIP_PATH"] + '/include',
      '-I' + self.path + '/../../external/picojson',
      '--std=c++17',
      '-c',
      self.path + '/' + self.file,
      ]
    # HIP compiler on AMD platforms has limit of 20 errors, and some negative
    # test cases expect that more errors are detected.
    if (self.platform == 'amd'):
      compiler_args.append('-ferror-limit=200')
    compiler_output = subprocess.run(compiler_args, stderr=subprocess.PIPE)
    # Get the compiler output in the stdout if -V flag is raised during ctest invocation.
    print(compiler_output.stderr.decode('UTF-8'))
    if self.expected_error_count < 0:
      self.assertGreater(compiler_output.stderr.decode('UTF-8').count(self.error_string), 0)
    else:
      self.assertEqual(compiler_output.stderr.decode('UTF-8').count(self.error_string),
                       self.expected_error_count)

if __name__ == '__main__':
  if len(sys.argv) == 5:
    CompileAndCapture.path = sys.argv[1]
    CompileAndCapture.platform = sys.argv[2]
    CompileAndCapture.file = sys.argv[3]
    CompileAndCapture.expected_error_count = int(sys.argv[4])
  # Unittest looks at the same argv's as the __main__ and doesn't know how
  # to handle arguments other than the executable (0). Therefore passing only
  # executable as the argv for unittest module.
  unittest.main(argv=[sys.argv[0]])
