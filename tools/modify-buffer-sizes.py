#!/usr/bin/env python3
"""
Script to calculate buffer sizes based on some custom elements added to the XMLModelFile schema? 

@author Peter Heywood <p.heywood@sheffield.ac.uk>
"""


import os
import sys
import math
import argparse
import xml.etree.ElementTree as ET
import re


# Dictionary of the expected XML namespaces, making search easier
NAMESPACES = {
  "xmml": "http://www.dcs.shef.ac.uk/~paul/XMML",
  "gpu":  "http://www.dcs.shef.ac.uk/~paul/XMMLGPU"
}

def verbose_log(args, msg):
  if args.verbose:
    print("Log: " + msg)

def cli():
    # Process command line arguments
    parser = argparse.ArgumentParser(
        description="Scale <bufferSize> values based on equations?"
    )
    parser.add_argument(
        "XMLModelFile",
        type=str,
        help="path to XMLModelFile to modify"
    )
    parser.add_argument(
        "-o",
        "--output",
        action="store_true",
        help="Output location"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    args = parser.parse_args()
    return args

def validate_args(args):
    valid = True
    if not os.path.isfile(args.XMLModelFile):
        print(f"Error: {args.XMLModelFile} does not exist.")
        valid = False

    if args.output is not None:
        if os.path.isdir(args.output):
            print(f"Error: {args.output} is a directory.")
            valid = False
        elif os.path.isfile(args.output):
            if not args.force:
                print(f"Error: {args.output} already exists, use -f/--force to overwrite.")
                valid = False

    return valid

def load_xmlmodelfile(args):
  verbose_log(args, "Loading XML Model File")

  # Attempt to parse the file. 
  tree = ET.parse(args.XMLModelFile)

  return tree

def get_bufferSizes(args, xmlroot):
    data = []
    for x in xmlroot.findall('.//gpu:bufferSize', NAMESPACES):
        data.append(x)
    return data

def ceil_pow2(x):
    # For a given value, round it up to the nearest power of 2 number.
    res = int(pow(2, math.ceil(math.log(x) / math.log(2))))
    return res

# @todo - don't use eval / validate the string beforehand
# @todo - build a variable dependency graph, by splitting math operators out of equations, to figure out which order to compute values in. 
def update_buffer_sizes(args, xml_in):
    verbose_log(args, "Generating new bufferSizes")

    # Get all existing buffer size tags
    buffer_size_tags = get_bufferSizes(args, xml_in)
    # Filter the buffer sizes into categories - raw values with ids, values to be computed with ids, values to be computed without ids.


    valid_id_pattern = re.compile("^[a-zA-Z_]{1}[a-zA-Z0-9_]*$")


    seen_buffer_size_ids = []
    buffer_sizes_id_only = {}
    buffer_sizes_id_and_compute = {}
    buffer_sizes_compute_only = []
    for buffer_size_tag in buffer_size_tags:
        # Get values from the xml.
        buffer_size_id = buffer_size_tag.get('id')
        buffer_size_compute = buffer_size_tag.get('relativeBufferSize')
        buffer_size_value = int(buffer_size_tag.text)

        # If there is a buffer size
        if buffer_size_id is not None:
            # Ensure that the ID is valid.
            if valid_id_pattern.match(buffer_size_id) is None:
                print(f"Error: id `{buffer_size_id}` is not valid as a python variable name.")
                sys.exit(1)

            # Append to the list of seen buffer sizes
            seen_buffer_size_ids.append(buffer_size_id)

            # If there is compute to do, 
            if buffer_size_compute is not None and len(buffer_size_compute) > 0:
                # Store in the list of id and compute
                buffer_sizes_id_and_compute[buffer_size_id] = {
                    "xml": buffer_size_tag,
                    "eqn": buffer_size_compute,
                    "val": None
                }
            else:
                # Store in the list of id onlys
                buffer_sizes_id_only[buffer_size_id] = {
                    "xml": buffer_size_tag,
                    "val": buffer_size_value,
                }
        elif buffer_size_compute is not None and len(buffer_size_compute) > 0:
            # If no ID, store in a list to be processed last.
                buffer_sizes_compute_only.append({
                    "xml": buffer_size_tag,
                    "eqn": buffer_size_compute,
                    "val": None
                })

    # Validate that ids were unique.
    total_buffer_size_ids = len(seen_buffer_size_ids)
    uniq_buffer_size_ids = len(set(seen_buffer_size_ids))
    if(total_buffer_size_ids != uniq_buffer_size_ids):
        print("Error: Non-unique <bufferSize id>")
        sys.exit(1)


    # Build a dictionary of computed values.
    computed_values = {}

    # Add each id only value to the computed values
    for k in buffer_sizes_id_only:
        computed_values[k] = buffer_sizes_id_only[k]["val"]

    # For each compute and id value, calculate and store.
    # @note - it is possible this would never complete if dependency circle - DAG validation would fix this.
    to_compute = list(buffer_sizes_id_and_compute.keys())
    # Worst case passes is the length of the array.
    worst_case_passes = len(to_compute)
    for iteration in range(worst_case_passes):
        # Prep a new dict.
        still_to_compute = []
        # Try to make each bit
        for k in to_compute:
            # If not yet calculated, calculate and store.
            if buffer_sizes_id_and_compute[k]["val"] is None:
                try:
                    eqn = buffer_sizes_id_and_compute[k]["eqn"]
                    eqn_result = eval(eqn, computed_values.copy())
                    rounded = ceil_pow2(eqn_result)
                    buffer_sizes_id_and_compute[k]["val"] = rounded
                    computed_values[k] = rounded
                except Exception as e:
                    still_to_compute.append(k) 
        to_compute = still_to_compute
        if len(to_compute) == 0:
            break
    if len(to_compute) > 0:
        for k in to_compute:
            eqn = buffer_sizes_id_and_compute[k]["eqn"]
            print("Error: Could not compute `{:}` from `{:}`. Invalid formula or dependency loop.".format(k, eqn))
        sys.exit(0)

    # Compute the result only ones, which cannot be a dependency from above so much simpler.
    for x in buffer_sizes_compute_only:
        if x["val"] is None:
            try:
                eqn = x["eqn"]
                eqn_result = eval(eqn, computed_values.copy())
                rounded = ceil_pow2(eqn_result)
                x["val"] = rounded
            except Exception as e:
                print("Error: Could not compute `{:}`. Invalid formula.".format(eqn))


    # Results are stored back in the buffer_sizes_id_and_compute and buffer_sizes_compute_only dict/lists
    # Update the XML and store values for formatted output.
    data = []
    for k, v in buffer_sizes_id_only.items():
        data.append([k, "", str(v["val"])])

    for k, v in buffer_sizes_id_and_compute.items():
        data.append([k, v["xml"].text, str(v["val"])])
        v["xml"].text = str(v["val"])

    for x in buffer_sizes_compute_only:
        data.append(["???", x["xml"].text, str(x["val"])])
        x["xml"].text = str(x["val"])


    # Print output.
    print("Buffer Sizes:")
    col_count = max([len(r) for r in data])
    col_dirs = [">" for x in range(col_count)]
    col_dirs[0] = "<"
    col_lengths = []
    for col in range(col_count):
        col_lengths.append(max([len(r[col]) for r in data]))
    for row in data:
        row_strings = []
        for col in range(col_count):
            row_strings.append("{0:{2}{1}}".format(row[col], col_lengths[col], col_dirs[col]))
        print(" ".join(row_strings))


    xml_out = xml_in
    return xml_out

def save_xml(args, xml):
    # Get destination
    f = args.XMLModelFile if not args.output else args.output
    # Log if verbose
    verbose_log(args, f"Saving new XML to disk as {f}")
    # Strip default namespace
    for ns_key, ns_address in NAMESPACES.items():
        if ns_key == "xmml":
            ns_key = ''
        ET.register_namespace(ns_key, ns_address)
    # Write to disk
    xml.write(f, encoding="utf-8", xml_declaration=True)

def main():
    args = cli()
    if not validate_args(args):
        return False
    xml = load_xmlmodelfile(args)
    updated_xml = update_buffer_sizes(args, xml)
    saved = save_xml(args, xml)


if __name__ == "__main__":
    main()
