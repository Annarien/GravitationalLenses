import csv
from csv import writer


def createExcelSheet(excel_file_path, headers):
    # excel_file_path is the full file path, with extension, to the file to write.
    # columns is an array of columns to add to the excel file.
    # overwrite defines whether it should overwrite the file if it exists or not. Default is false.

    with open(excel_file_path, 'w', newline='') as out_csv:
        writer = csv.DictWriter(out_csv, headers)
        writer.writeheader()

    print("Creating excel file at: %s" % excel_file_path)


def writeToFile(excel_file_path, element_dictionary):
    # excel_file_path is the full file path to the excel file.
    #      Call createExcelSheet with overwrite=False here just in case it doesn't exist.
    # data is an array of data to fill the row with.
    # columns is an array of columns. This can be null, but if not then len(data) must equal len(columns)
    # This is the only definition you should need to import. The one above becomes a helper hidden in the background.

    with open(excel_file_path, 'a+') as write_object:
        csv_writer = writer(write_object)
        csv_writer.writerow(element_dictionary)
