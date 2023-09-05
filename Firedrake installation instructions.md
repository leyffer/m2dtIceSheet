# Firedrake download page

The firedrake download page outlines the installation instructions: [https://www.firedrakeproject.org/download.html]

## Basic steps

1. Download the installation script:

   ```bash
   curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
   ```

2. If anaconda/miniconda/etc. is installed, you will need to comment out any code from `~/.bashrc`/`~/.zshrc`/etc. between the `# >>> conda initialize >>>` and `# <<< conda initialize <<<`.
3. Run the installation script:

   ```bash
   python3 firedrake-install
   ```

4. When I had to reinstall firedrake, I had trouble with the installation script (the creation of the virtual environment was not successful). This required modifying the script to skip the creation of the virtual environment (line 1719) and creating the virtual environment separately using

   ```bash
   python3.11 -m venv firedrake
   ```

   The installation script needs to also be modified to not quit when encountering the exisitng virtual environment (Line 1705). Hopefully you will not have this problem at all (my first installation had no issues).
5. Activate the virtual environment by using

   ```bash
   source firedrake/bin/activate
   ```
