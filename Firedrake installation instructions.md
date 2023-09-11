# Firedrake download page

The firedrake download page outlines the installation instructions: [https://www.firedrakeproject.org/download.html]

## Basic steps

1. Download the installation script:

   ```bash
   curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
   ```

2. If anaconda/miniconda/etc. is installed, you will need to comment out any code from `~/.bashrc`/`~/.zshrc`/etc. between the `# >>> conda initialize >>>` and `# <<< conda initialize <<<`. This can (probably) be uncommented after installation.
3. Run the installation script:

   ```bash
   python3 firedrake-install
   ```

   The installation script can be customized. Take a look at the help if you would like to customize the installation:
   
   ```bash
   python3 firedrake-install --help
   ```

   The installation script will create a directory `firedrake` containing a virtual environment within the directory it is in.

4. When I had to reinstall firedrake, I had trouble with the installation script (the creation of the virtual environment was not successful due to a problem with the `venv` creation call). This required modifying the script to skip the creation of the virtual environment (line 1719) and creating the virtual environment separately using

   ```bash
   python3.11 -m venv firedrake
   ```

   The installation script needs to also be modified to not quit when encountering the exisitng virtual environment (Line 1705). **Hopefully you will not have this problem at all (my first installation had no issues)**. The modified installation script section (starting at line 1702) is as follows:

   ```python
   if mode == "install":
       if os.path.exists(firedrake_env):
              log.warning("Specified venv '%s' already exists", firedrake_env)
              # quit("Can't install into existing venv '%s'" % firedrake_env) # We now allow for continuing if the venv already exists.
              os.environ["VIRTUAL_ENV"] = firedrake_env
       else:
           log.info("Creating firedrake venv in '%s'." % firedrake_env)
           # Debian's Python3 is screwed, they don't ship ensurepip as part
           # of the base python package, so the default python -m venv
           # doesn't work.  Moreover, they have spiked the file such that it
           # calls sys.exit, which will kill any attempts to create a venv
           # with pip.
           try:
               import ensurepip        # noqa: F401
               with_pip = True
           except ImportError:
               with_pip = False
           import venv
           venv.EnvBuilder().create(firedrake_env) # Problematic line for my installation
           if not with_pip:
               import urllib.request
               log.debug("ensurepip unavailable, bootstrapping pip using get-pip.py")
               urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", filename="get-pip.py")
               check_call([python, "get-pip.py"])
               log.debug("bootstrapping pip succeeded")
               log.debug("Removing get-pip.py")
               os.remove("get-pip.py")
   
           # We haven't activated the venv so we need to manually set the environment.
           os.environ["VIRTUAL_ENV"] = firedrake_env
   ```

5. Activate the virtual environment by using

   ```bash
   source firedrake/bin/activate
   ```

   or

   ```bash
   . firedrake/bin/activate
   ```
