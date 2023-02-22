"""
Usage: python3 extract_people_df.py <matsim_output.xml>

Clément Dauvilliers 13/03/2022 - EPFL TRANSP-OR lab semester project

Extracts all information regarding individuals in the synthetic population in the
MATSIM output for Switzerland.
"""
import xml.etree.ElementTree as ET
import sys
import csv
import os
import gzip
import json
from optparse import OptionParser
from time import time

# PARAMETERS
# Whether to extract the activities
_PROCESS_ACTIVITIES_ = True
# Whether to extract the attributes of people on top of the activities
_PROCESS_PERSON_ATTRIBUTES_ = True
# Whether to extract the transport information
_PROCESS_TRANSPORTS_ = True
# Save file paths
_SAVE_FILE_ATTRIBUTES_ = "data/extracted_data/matsim_population_attributes.csv"
_SAVE_FILE_ACTIVITIES_ = "data/extracted_data/matsim_population_activities.csv"
_SAVE_FILE_TRANSPORTS_ = "data/extracted_data/matsim_population_transports.csv"

_FIELDS_ATTRIBUTES_ = ['id', 'age', 'bikeAvailability', 'carAvail', 'employed',
                       'hasLicense', 'home_x', 'home_y', 'householdIncome',
                       'isCarPassenger', 'municipalityType', 'ptHasGA', 'ptHasHalbtax',
                       'ptHasStrecke', 'ptHasVerbund', 'sex', 'spRegion']

_FIELDS_ACTIVITIES_ = ['id', 'type', 'facility', 'link', 'x', 'y', 'start_time', 'end_time']
_FIELDS_TRANSPORTS_ = ['id', 'mode', 'dep_time', 'trav_time', 'start_link', 'end_link', 'transitRouteId',
                       'boardingTime', 'transitLineId', 'accessFacilityId', 'egressFacilityId']


def refresh_xml_parser(current_parser=None, add_xml_root=False):
    """
    Recreates a new XML parser to free all XML elements stored
    in the current one.
    :param current_parser: the current XML parser being used.
        if None, simply creates a new parser. If not None, sets add_xml_root
        to True automatically.
    :param add_xml_root: Boolean, whether to artificially add the XML
        root node. This should be set to True if the new parser will not
        read the XML from the beginning, for any reason.
    :return: a fresh new XML parser. The new parser has ALREADY
        read the header lines of the document (from the <?xml tag
        to the <population>). It is meant to be directly fed
        <person> elements.
    """
    # Create a new parser.
    new_parser = ET.XMLPullParser(events=('end',))
    if current_parser is not None:
        # Delete the current parser. As it hasn't finished reading
        # the XML file, it is expecting to see a </population> and </xml>
        # tags. Thus the close() method will raise an exception, which we
        # actually don't mind.
        try:
            current_parser.close()
        except ET.ParseError:
            print("Recreating XML parser")
        add_xml_root = True

    if add_xml_root:
        # Unfortunately, if we directly give the
        # next lines of the XML files (new <person> elements), the parser will raise
        # an exception as it expects a root (like <?xml>).
        # The solution is to feed the new parser with the roots elements (the first few
        # lines of the document) until the <population> tag so that it is ready
        # to receive new <person> elements.
        document_root = '<?xml version="1.0" encoding="utf-8"?>\
        <!DOCTYPE population SYSTEM "http://www.matsim.org/files/dtd/population_v6.dtd">\
        <population desc="Switzerland Baseline">'
        new_parser.feed(document_root)
    return new_parser


def create_csv_file(path, fieldnames, recreate_file=False):
    """
    Creates a CSV file at the given path and writes
    its header. If the file already exists, does
    nothing.
    :param path: where to create the csv;
    :param fieldnames: list of names.
    :param recreate_file: Boolean, whether to re-create the file or to append to it.
    :return: (file, writer) where:
        - file is the python File object used to write into the file.
        - writer is the csv.DictWriter object used to write the CSV.
    """
    if not recreate_file:
        file = open(path, "a", newline='')
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction='ignore')
    else:
        file = open(path, "w", newline='')
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
    return file, writer


def fetch_leg_info(leg, person_id):
    """
    Given a <leg> object, retrieves its valuable information
    and stores it into a dictionnary.
    """
    leg_dict = dict()
    # Attributes that are common to all transport modes
    leg_dict['id'] = person_id
    leg_dict['mode'] = leg.attrib['mode']
    leg_dict['dep_time'] = leg.attrib['dep_time']
    leg_dict['trav_time'] = leg.attrib['trav_time']
    route = leg[1]
    leg_dict['start_link'] = route.attrib['start_link']
    leg_dict['end_link'] = route.attrib['end_link']
    # Attributes for public transports: they're stored in the text of the
    # <route> element, in JSON format
    if leg_dict['mode'] == 'pt':
        leg_pt_attribs = json.loads(route.text)
        leg_dict.update(leg_pt_attribs)

    return leg_dict


def main():
    # PARSING OPTIONS
    optparser = OptionParser()
    optparser.add_option('--chunksize', action="store", dest="chunksize", type="int", default=100000,
                         help="Number of agents to process before saving and cleaning the RAM. Default is 100,000,"
                              " which requires about 6GB of memory.")
    optparser.add_option('--first_line', action="store", dest="first_line", type="int", default=0,
                         help="First line of the XML to parse. Default: 0. Use this option if the"
                              " program was stopped before finishing.")
    (options, args) = optparser.parse_args()

    if len(args) < 1:
        print("Usage: python3 extract_people_df.py <xml_file_path>")
        sys.exit(-1)

    # Counts the number of individuals that have been processed
    processed_people = 0

    # Creates the save files if we process the XML from the start
    # Otherwise, append the newly processed lines to it
    create_files = options.first_line == 0
    attributes_csv, attributes_writer = create_csv_file(_SAVE_FILE_ATTRIBUTES_, _FIELDS_ATTRIBUTES_, create_files)
    activities_csv, activities_writer = create_csv_file(_SAVE_FILE_ACTIVITIES_, _FIELDS_ACTIVITIES_, create_files)
    transports_csv, transports_writer = create_csv_file(_SAVE_FILE_TRANSPORTS_, _FIELDS_TRANSPORTS_, create_files)

    # Lists containing the last rows that have been extracted from the XML
    # but have yet to be written into the CSV file
    soc_eco_attrib_rows = []
    last_activities = []
    last_transports = []

    start_time = time()
    with gzip.open(args[0], "r") as matsim_file:
        # Builds the XML non-blocking parser, which will be fed
        # with the successive lines from the matsim file.
        # If we don't read from line 0, then we must artificially add the XML root node
        # to the parser or it will raise an exception.
        xml_parser = refresh_xml_parser(add_xml_root=options.first_line > 0)
        for line_number, line in enumerate(matsim_file):
            if line_number < options.first_line:
                continue

            xml_parser.feed(line)
            for event, elem in xml_parser.read_events():
                # We browse the XML until we find the end of a <person> element.
                # Then, we retrieve the list of all its attributes
                # and store their names and values
                if event == "end" and elem.tag == "person":
                    # We do not count people whose ids are 'freight_xxxxx' as we do not
                    # have socio-economic information about them
                    person_id = elem.attrib['id']
                    if person_id[0] == 'f':
                        continue
                    processed_people += 1

                    # ==== PROCESSING ACTIVITIES =========== #
                    # We have to find the <plan> element, which contains the info about activities and transport.
                    # The plan is usually given by elem[1].
                    # Sometimes, there are multiple <plan> elements for a person, but only one is set as "selected".
                    # In this case, we should skip the not selected plans until finding the right one.
                    for potential_plan in elem[1:]:
                        if potential_plan.attrib['selected'] == 'yes':
                            plan = potential_plan
                            break
                    if _PROCESS_ACTIVITIES_:
                        activities = [xmlelem for xmlelem in plan if xmlelem.tag[0] == 'a']
                        # Compiles the attributes (type, facility, x, y, ...) of all activities
                        # of the current person.
                        person_activities = [
                            {name: value for name, value in activity.attrib.items()}
                            for activity in activities
                        ]
                        # Adds the ID of the current person to that information, then
                        # add all of it to last_activities so that they'll be written into the
                        # results file
                        for activity_data in person_activities:
                            activity_data['id'] = person_id
                        last_activities += person_activities

                    # ==== PROCESSING ATTRIBUTES ========== #
                    if _PROCESS_PERSON_ATTRIBUTES_:
                        # The children elements of a <person> are <attributes> and <plan>
                        # The <plan> concerns the activities, which we ignore here
                        attributes = list(elem[0])
                        # att.text contains the value of the attribute
                        # such as "45" for age, or "true" for employed.
                        attributes = {att.attrib['name']: att.text for att in attributes}
                        attributes['id'] = person_id
                        soc_eco_attrib_rows.append(attributes)
                        # Special case just for fun: sometimes,
                        # the home location is missing from the attributes. This seems to happen
                        # when the individual either stays at home all day, or visits multiple 'home' locations,
                        # making MATSIM unable to decide which one is right.
                        # We can still assume that the first 'home' address visited is the right one in most cases.
                        if 'home_x' not in attributes and _PROCESS_ACTIVITIES_:
                            for activity in person_activities:
                                if activity['type'] == 'home':
                                    attributes['home_x'] = activity['x']
                                    attributes['home_y'] = activity['y']
                                    break

                    # Same treatment for the legs (transportation) as for the activities
                    if _PROCESS_TRANSPORTS_:
                        # Extracts the <leg> tags of the current person
                        legs = [xmlelem for xmlelem in plan if xmlelem.tag == 'leg']
                        # Fetches the information about each leg
                        last_transports += [fetch_leg_info(leg, person_id) for leg in legs]

                    if processed_people % 10000 == 0:
                        print(f'Processed {processed_people} people in {time() - start_time}s')

                    # After a fix number of individuals processed, saves their attributes
                    # and activities
                    if processed_people % options.chunksize == 0:
                        # Write the attributes
                        if _PROCESS_PERSON_ATTRIBUTES_:
                            print(f'Saving attributes to {_SAVE_FILE_ATTRIBUTES_}, last line processed={line_number}')
                            attributes_writer.writerows(soc_eco_attrib_rows)
                        # Write the activities
                        if _PROCESS_ACTIVITIES_:
                            activities_writer.writerows(last_activities)
                            print(f'Saving activities to {_SAVE_FILE_ACTIVITIES_}, last line processed={line_number}')
                        # Write the transports
                        if _PROCESS_TRANSPORTS_:
                            print(f'Saving transports to {_SAVE_FILE_TRANSPORTS_}, last line processed={line_number}')
                            transports_writer.writerows(last_transports)

                        # Frees all memory used so far to store attributes and activities
                        del soc_eco_attrib_rows
                        del last_activities
                        del last_transports
                        soc_eco_attrib_rows = []
                        last_activities = []
                        last_transports = []
                        # Recreates the XML parser to free the memory, otherwise
                        # all XML seen so far is still remembered and kept in mem
                        xml_parser = refresh_xml_parser(xml_parser)

    # Close the CSV files before exiting
    activities_csv.close()
    attributes_csv.close()
    transports_csv.close()
    return 0


if __name__ == "__main__":
    main()
