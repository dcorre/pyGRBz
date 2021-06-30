#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#  Original file comes from sncosmo
"""Functions for reading GRB light curves.
   Adapted from Python module 'sncosmo'"""

from __future__ import print_function

from collections import OrderedDict as odict

import numpy as np
from astropy.table import Table, vstack
import six

__all__ = [
    "read_lc",
    "load_observations",
    "load_info_observations",
    "load_telescope_transmissions",
]


def _cast_str(s):
    try:
        return int(s)
    except:
        try:
            return float(s)
        except:
            return s.strip()


# -----------------------------------------------------------------------------
# Reader: ascii
def _read_ascii(f, **kwargs):

    delim = kwargs.get("delim", None)
    metachar = kwargs.get("metachar", "@")
    commentchar = kwargs.get("commentchar", "#")

    meta = odict()
    colnames = []
    cols = []
    readingdata = False
    for line in f:

        # strip leading & trailing whitespace, newline, and comments
        line = line.strip()
        # Find name, format and time in comment
        if len(line) > 0 and line[0] == "#":
            if line[1] == "#":
                pass  # comment line
            pos = line.find(":")
            if pos in [-1, 1]:
                pass  # comment line
            if line[1:pos].strip().lower() == "name":
                grb_name = str(line[pos + 1 :].strip())
            if line[1:pos].strip().lower() == "type":
                grb_format = str(line[pos + 1 :].strip())
            if line[1:pos].strip().lower() == "time_since_burst":
                time = str(line[pos + 1 :].strip())

        pos = line.find(commentchar)
        if pos > -1:
            line = line[:pos]
        if len(line) == 0:
            continue

        if not readingdata:
            # Read metadata
            if line[0] == metachar:
                pos = line.find(" ")  # Find first space.
                if pos in [-1, 1]:  # Space must exist and key must exist.
                    raise ValueError("Incorrectly formatted metadata line: " + line)
                meta[line[1:pos]] = _cast_str(line[pos:])
                continue

            # Read header line
            if grb_format.lower() == "lc":
                colnames.extend(["Name"])
            elif grb_format.lower() == "sed":
                colnames.extend(["Name", "time_since_burst"])
            for item in line.split(delim):
                colnames.append(item.strip())
                cols.append([])
            # add columns for the grb name and time since burst
            cols.extend([[], []])
            readingdata = True
            continue
        # Now we're reading data
        items = []
        items.append(grb_name)
        if grb_format.lower() == "sed":
            items.extend([time])
        items.extend(line.split(delim))
        for col, item in zip(cols, items):
            # print ('col: {}'.format(col))
            # print ('items: {}'.format(items))
            col.append(_cast_str(item))

    data = odict(zip(colnames, cols))
    return meta, data


# -----------------------------------------------------------------------------
# All readers
READERS = {"ascii": _read_ascii}


def read_lc(file_or_dir, format="ascii", **kwargs):
    """Read light curve data for a single supernova.

    Parameters
    ----------
    file_or_dir : str
        Filename (formats 'ascii', 'salt2') or directory name
        (format 'salt2-old'). For 'salt2-old' format, directory must contain
        a file named 'lightfile'. All other files in the directory are
        assumed to be photometry files, unless the `filenames` keyword argument
        is set.
    format : {'ascii', 'salt2', 'salt2-old'}, optional
        Format of file. Default is 'ascii'. 'salt2' is the new format available
        in snfit version >= 2.3.0.
    delim : str, optional
        **[ascii only]** Used to split entries on a line. Default is `None`.
        Extra whitespace is ignored.
    metachar : str, optional
        **[ascii only]** Lines whose first non-whitespace character is
        `metachar` are treated as metadata lines, where the key and value
        are split on the first whitespace. Default is ``'@'``
    commentchar : str, optional
        **[ascii only]** One-character string indicating a comment. Default is
        '#'.
    filenames : list, optional
        **[salt2-old only]** Only try to read the given filenames as
        photometry files. Default is to try to read all files in directory.

    Returns
    -------
    t : astropy `~astropy.table.Table`
        Table of data. Metadata (as an `OrderedDict`) can be accessed via
        the ``t.meta`` attribute. For example: ``t.meta['key']``. The key
        is case-sensitive.

    Examples
    --------

    Read an ascii format file that includes metadata (``StringIO``
    behaves like a file object):

    >>> from astropy.extern.six import StringIO
    >>> f = StringIO('''
    ... @id 1
    ... @RA 36.0
    ... @description good
    ... time band flux fluxerr zp zpsys
    ... 50000. g 1. 0.1 25. ab
    ... 50000.1 r 2. 0.1 25. ab
    ... ''')
    >>> t = read_lc(f, format='ascii')
    >>> print(t)
      time  band flux fluxerr  zp  zpsys
    ------- ---- ---- ------- ---- -----
    50000.0    g  1.0     0.1 25.0    ab
    50000.1    r  2.0     0.1 25.0    ab
    >>> t.meta
    OrderedDict([('id', 1), ('RA', 36.0), ('description', 'good')])

    """

    try:
        readfunc = READERS[format]
    except KeyError:
        raise ValueError(
            "Reader not defined for format {0!r}. Options: ".format(format)
            + ", ".join(READERS.keys())
        )

    if format == "salt2-old":
        meta, data = readfunc(file_or_dir, **kwargs)
    elif isinstance(file_or_dir, six.string_types):
        with open(file_or_dir, "r", encoding="utf-8") as f:
            meta, data = readfunc(f, **kwargs)
    else:
        meta, data = readfunc(file_or_dir, **kwargs)

    return Table(data, meta=meta, masked=True)


def load_observations(filenames):
    """Load observations, either a light curve or sed

    Returns
    -------
    data: astropy Table

    """
    #  Use vstack only at the end, otherwise memory usage explodes
    data_list = []
    for counter, filename in enumerate(filenames):
        data = read_lc(filename, format="ascii")
        data_list.append(data)
        # if counter==0: data_tot=data
        # else: data_tot=vstack([data_tot,data],join_type='inner')
    # data_tot=vstack(data_list,join_type='inner')
    data_tot = vstack(data_list)
    return data_tot


def load_info_observations(filenames):
    """Load observations, either a light curve or sed

    Returns
    -------
    data: astropy Table

    """
    # name:GRB130606A
    # zspec:5.913
    # zsim:5.90
    # zim_sup:0.29
    # zsim_inf:0.22
    # Av:0.13
    # dust_type:SMC
    # beta:0.42
    data_list = []
    for counter, filename in enumerate(filenames):
        references = []
        values = []
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                # strip leading & trailing whitespace, newline, and comments
                line = line.strip()
                if len(line) == 0:
                    continue
                # Read header line
                if line[0] == "#":
                    if line[1] == "#":
                        continue  # comment line
                    pos = line.find(":")

                    if pos in [-1, 1]:
                        continue  # comment line
                    ref = line[1:pos].strip()
                    references.append(ref)
                    val = line[pos + 1 :].strip()
                    values.append(val)
        data = Table(np.array(values), names=list(references))
        data_list.append(data)
        # if counter==0: data_info = data
        # else: data_info = vstack([data_info,data],join_type='outer')
    data_info = vstack(data_list, join_type="outer")
    # sort by name
    data_info.sort("name")

    return data_info


def load_telescope_transmissions(
    info_dict, wavelength, norm=False, norm_val=1.0, resamp=True
):
    """
    Load the transmittance of the selected band of the selected
    telescope with respect to wavelength

    Parameters
    ----------
    info_dict: dictionary

    wavelength : array
        wavelengths in angstrom

    norm: boolean
        enables to normalise the values to 'norm_val' (default: False)

    norm_val: float
        value used for normalising the data

    Returns
    ---------
    trans :  array
        transmittance of the mirror at a given wavelength  (0-1)
    """
    from .utils import resample

    filter_path = (
        info_dict["path"]
        + "/transmissions/"
        + info_dict["telescope"]
        + "/"
        + info_dict["band"]
        + ".txt"
    )
    File = open(filter_path, "r")
    lines = File.readlines()

    wvl = []
    trans = []

    for line in lines:
        if line[0] != "#" and len(line) > 3:
            bits = line.split()
            trans.append(float(bits[1]))
            wvl.append(float(bits[0]))

    wvl = np.array(wvl) * 10.0  # nm --> angstroms
    if max(trans) > 1:
        trans = np.array(trans, dtype=np.float64) * 1e-2

    # Normalisation
    if norm:
        trans = trans / max(trans) * norm_val

    if resamp:
        # Resample the transmission to the
        trans = resample(wvl, trans, wavelength, 0.0, 1.0)
    return wvl, trans
