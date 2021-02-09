from ipykernel import connect_qtconsole
from ipykernel.kernelapp import IPKernelApp


def mpl_kernel(gui='qt'):
    """
    Launch and return an IPython kernel with matplotlib support for the desired gui
    """
    kernel = IPKernelApp.instance()
    kernel.initialize(['python', '--matplotlib=%s' % gui])
    # kernel.initialize(['python'])
    return kernel


class InternalIPKernel(object):
    def __init__(self) -> None:
        self.ipkernel = None
        # To create and track active qt consoles
        self.namespace = {}

    def init_ipkernel(self, backend='qt'):
        # Start IPython kernel with GUI event loop and mpl support
        self.ipkernel = mpl_kernel(backend)

        # This application will also act on the shell user namespace
        self.namespace = self.ipkernel.shell.user_ns

    def new_qt_console(self, evt=None):
        """start a new qtconsole connected to our kernel"""
        new_console = connect_qtconsole(self.ipkernel.abs_connection_file,
                                        profile=self.ipkernel.profile)
        return new_console
