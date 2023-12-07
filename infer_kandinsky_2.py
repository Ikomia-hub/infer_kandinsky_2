from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits PyDataProcess.CPluginProcessInterface from Ikomia API
# --------------------
class IkomiaPlugin(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def get_process_factory(self):
        # Instantiate algorithm object
        from infer_kandinsky_2.infer_kandinsky_2_process import InferKandinsky2Factory
        return InferKandinsky2Factory()

    def get_widget_factory(self):
        # Instantiate associated widget object
        from infer_kandinsky_2.infer_kandinsky_2_widget import InferKandinsky2WidgetFactory
        return InferKandinsky2WidgetFactory()
