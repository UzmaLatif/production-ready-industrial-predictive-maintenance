class PredictiveMaintenanceException(Exception):
    """
    Custom exception for the Predictive Maintenance pipeline.
    Gives clear error messages showing exactly which file 
    and line number caused the error.
    """
    def __init__(self, error_message: str, error_detail):
        super().__init__(error_message)
        
        # Extract file name and line number from error detail
        _, _, exc_tb = error_detail.exc_info()
        self.line_number = exc_tb.tb_lineno
        self.file_name = exc_tb.tb_frame.f_code.co_filename
        
        self.error_message = (
            f"Error in file: [{self.file_name}] "
            f"at line: [{self.line_number}] "
            f"— {error_message}"
        )

    def __str__(self):
        return self.error_message