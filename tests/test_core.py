# Copyright 2025 Paras (PR00T - Paras Robotics 00 Technology)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from pr00t.core import pr00t_main


def test_pr00t_main(capsys):
    """
    Test the pr00t_main function to ensure it runs without errors and produces expected output.
    """
    pr00t_main()

    # Capture the output
    captured = capsys.readouterr()

    # Check if the welcome message is in the output
    assert "Welcome to PR00T!" in captured.out
    assert "PR00T is running..." in captured.out
