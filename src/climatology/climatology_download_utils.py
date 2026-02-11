import subprocess
from datetime import datetime

def get_subfolders():
    
        valid_subfolders = subprocess.run(
            ["rclone", "lsf", f"climatex2:/Share_Data/"],
            capture_output=True,
            text=True,
            check=True,
        )

        climatex_subfolders = subprocess.run(
            ["rclone", "lsf", f"climatex:/Share_Data/"],
            capture_output=True,
            text=True,
            check=True,
        )
        valid_subfolders.stdout += climatex_subfolders.stdout
        return valid_subfolders

def get_water_year_start_from_month(date):
    """
    Water year folders are prefixed with WPS_YYYYMM_YYYYMM_
    Get the folder prefix based on which water year the date falls into
    date: datetime.datetime
    """
    month = date.month
    year = date.year
    first_day_condition = ((date.day == 1) and (date.hour == 0) and (month == 10)) # first day of first month at T00:00:00 => take year before
    if (month >= 10) and not first_day_condition:
        start_year = date.year
        end_year = date.year + 1
    else:
        start_year = date.year - 1
        end_year = date.year

    return start_year, end_year
    
def get_water_year_folder_prefix(date):
    """
    Water year folders are prefixed with WPS_YYYYMM_YYYYMM_
    Get the folder prefix based on which water year the month_date falls into
    date: datetime.datetime
    """
    start_year, end_year = get_water_year_start_from_month(date)
    water_year_start_str = f"{start_year}09"
    water_year_end_str = f"{end_year}12"
    folder_prefix = f"WPS_{water_year_start_str}_{water_year_end_str}"

    return folder_prefix

def find_water_year_subfolder(date, valid_subfolders):
        """
        Find the subfolder for the water year by searching valid_subfolders
        e.g.: 1989-10-01 -> WPS_198909_199012_{user}
        date: datetime.datetime
        """
        folder_prefix = get_water_year_folder_prefix(date)

        folders = [
            line
            for line in valid_subfolders.stdout.splitlines()
            if line.startswith(folder_prefix) and (len(line.split("_")) == 4)
        ]
        if not folders:
            raise FileNotFoundError(f"No folder found matching prefix {folder_prefix}")

        # filter down empty / confusing folders

        folder = folders[0].rstrip("/")
        user = folder.split("_")[-1]

        return folder

def get_rclone_source(year):
    if (year >= 2018) and (year < 2023):
        rclone_source = "climatex"
    else:
        rclone_source = "climatex2"
        print(f" ⛴️ rclone source: {rclone_source}", flush=True)
    return rclone_source