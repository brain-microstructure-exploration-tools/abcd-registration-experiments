import os
import numpy as np
import vtk

# From https://github.com/rordenlab/TractographyFormat/blob/master/PYTHON/read_mrtrix_tracks.py
def read_tck_header(in_file):
    fileobj = open(in_file, "rb")
    header = {}
    #iflogger.info("Reading header data...")
    for line in fileobj:
        line = line.decode()
        if line == "END\n":
            #iflogger.info("Reached the end of the header!")
            break
        elif ": " in line:
            line = line.replace("\n", "")
            line = line.replace("'", "")
            key = line.split(": ")[0]
            value = line.split(": ")[1]
            header[key] = value
            #iflogger.info('...adding "%s" to header for key "%s"', value, key)
    fileobj.close()
    header["count"] = int(header["count"].replace("\n", ""))
    header["offset"] = int(header["file"].replace(".", ""))
    return header

# From https://github.com/rordenlab/TractographyFormat/blob/master/PYTHON/read_mrtrix_tracks.py
def read_tck_streamlines(in_file, header):
    byte_offset = header["offset"]
    stream_count = header["count"]
    datatype = header["datatype"]
    dt = 4
    if datatype.startswith( 'Float64' ):
        dt = 8
    elif not datatype.startswith( 'Float32' ):
        print('Unsupported datatype: ' + datatype)
        return
    #tck format stores three floats (x/y/z) for each vertex
    num_triplets = (os.path.getsize(in_file) - byte_offset) // (dt * 3)
    dt = 'f' + str(dt)
    if datatype.endswith( 'LE' ):
        dt = '<'+dt    
    if datatype.endswith( 'BE' ):
        dt = '>'+dt
    vtx = np.fromfile(in_file, dtype=dt, count=(num_triplets*3), offset=byte_offset)
    vtx = np.reshape(vtx, (-1,3)) 
    #make sure last streamline delimited...
    if not np.isnan(vtx[-2,1]):
        vtx[-1,:] = np.nan
    line_ends, = np.where(np.all(np.isnan(vtx), axis=1));
    if stream_count != line_ends.size:
        print('expected {} streamlines, found {}'.format(stream_count, line_ends.size))
    line_starts = line_ends + 0
    line_starts[1:line_ends.size] = line_ends[0:line_ends.size-1]
    #the first line starts with the first vertex (index 0), so preceding NaN at -1
    line_starts[0] = -1;
    #first vertex of line is the one after a NaN
    line_starts = line_starts + 1
    #last vertex of line is the one before NaN
    line_ends = line_ends - 1
    return vtx, line_starts, line_ends

# From https://github.com/rordenlab/TractographyFormat/blob/master/PYTHON/read_mrtrix_tracks.py
def read_tck(in_file):
    header = read_tck_header(in_file)
    vertices, line_starts, line_ends = read_tck_streamlines(in_file, header)
    return header, vertices, line_starts, line_ends

# Modified from https://mail.python.org/pipermail/neuroimaging/2019-July/002006.html
def get_vtk_polydata(vertices, line_starts, line_ends):

    polydata = vtk.vtkPolyData()

    lines = vtk.vtkCellArray()
    points = vtk.vtkPoints()

    point_counter = 0

    for i in range(0, len(line_ends)):

        point_indices = np.arange(line_starts[i], line_ends[i]+1)
        num_points = len(point_indices)
        line = vtk.vtkLine()

        line.GetPointIds().SetNumberOfIds(num_points)

        for j in range(0, num_points):

            points.InsertNextPoint(vertices[point_indices[j], :])
            linePts = line.GetPointIds()
            linePts.SetId(j,point_counter)

            point_counter+=1

        lines.InsertNextCell(line)

    polydata.SetLines(lines)
    polydata.SetPoints(points)

    return polydata

def write_vtk(vertices, line_starts, line_ends, out_filename):

    polydata = get_vtk_polydata(vertices, line_starts, line_ends)
    
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(out_filename)
    writer.SetInputData(polydata)
    writer.Write()