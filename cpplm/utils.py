def parse_output(output):
    str_remove_spc = output.replace("<startofstring>", "").replace("<endofstring>", "")
    # remove content after the first <pad> token
    # print("#"*10, str_remove_spc)
    ret = str_remove_spc.split("<pad>")[0]
    # print("*"*10, ret)
    return ret
