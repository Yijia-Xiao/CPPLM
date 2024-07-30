def parse_output(output):
    str_remove_spc = output.replace("<startofstring>", "").replace("<endofstring>", "")
    # remove content after the first <pad> token
    ret = str_remove_spc.split("<pad>")[0]
    return ret
