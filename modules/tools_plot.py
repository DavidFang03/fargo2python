def get_coords_for_arrow(pos, delta, txt_dx, txt_dy):
    x, y = pos
    dx, dy = delta
    xytext = (x,y)
    xy = (x+dx, y+dy)
    postext = (x+dx+txt_dx, y+dy+txt_dy)
    return {"xy":xy, "xytext": xytext, "postext":postext}
