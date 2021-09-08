from PIL import Image, ImageDraw

import numpy as np

class Display():
    def __init__(self):
        self.image = Image.new("RGB", (500,500), (155,155,155))

    def drawEmptyCircle(self, x, z, r):
        draw = ImageDraw.Draw(self.image)
        draw.ellipse((250 + x - r, 250 - z - r, 250 + x + r, 250 - z + r), fill = "brown", outline="brown")
        return draw

    def drawEmptyRectangle(self, x, z, r1, r2, angle, multi):
        draw = ImageDraw.Draw(self.image)
        x1 = x + (r1) * np.cos(angle) - (r2) * np.sin(angle)
        y1 = z + (r2) * np.cos(angle) + (r1) * np.sin(angle)
        x2 = x + (-r1) * np.cos(angle) - (r2) * np.sin(angle)
        y2 = z + (r2) * np.cos(angle) + (-r1) * np.sin(angle)
        x3 = x + (-r1) * np.cos(angle) - (-r2) * np.sin(angle)
        y3 = z + (-r2) * np.cos(angle) + (-r1) * np.sin(angle)
        x4 = x + (r1) * np.cos(angle) - (-r2) * np.sin(angle)
        y4 = z + (-r2) * np.cos(angle) + (r1) * np.sin(angle)
        x1 = 250 + x1 * multi
        x2 = 250 + x2 * multi
        x3 = 250 + x3 * multi
        x4 = 250 + x4 * multi
        y1 = 250 + y1 * multi
        y2 = 250 + y2 * multi
        y3 = 250 + y3 * multi
        y4 = 250 + y4 * multi
        draw.polygon(xy = ((x1,y1), (x2,y2), (x3,y3), (x4,y4)), fill = "brown", outline="brown")
        return draw

    def angle(self, x, z, x1, z1):
        cos = ((x1 - x)*x + (z1 - z)*(z + 1)) / (np.sqrt(x*x + (z + 1)**2)*np.sqrt((x1 - x)**2 + (z1-z)**2))
        return np.arccos(cos)

    def drawSensorLine(self, x, z, mass, r1, r2, lenRay, angle, multi):
        draw = ImageDraw.Draw(self.image)
        if mass[0] == 1000:
            x1 = x + (-r1) * np.cos(angle)
            y1 = z + (-r1) * np.sin(angle)
            x2 = x + (-r1 - lenRay) * np.cos(angle)
            y2 = z + (-r1 - lenRay) * np.sin(angle)
            x1 = 250 + x1 * multi
            x2 = 250 + x2 * multi
            y1 = 250 + y1 * multi
            y2 = 250 + y2 * multi
            draw.line(xy = ((x1,y1), (x2,y2)), fill = 'brown')
        else:
            x1 = x + (-r1) * np.cos(angle)
            y1 = z + (-r1) * np.sin(angle)
            x2 = x + (-r1 - lenRay * mass[0] / 1000) * np.cos(angle)
            y2 = z + (-r1 - lenRay * mass[0] / 1000) * np.sin(angle)
            x1 = 250 + x1 * multi
            x2 = 250 + x2 * multi
            y1 = 250 + y1 * multi
            y2 = 250 + y2 * multi
            draw.line(xy = ((x1,y1), (x2,y2)), fill = 'brown')
            draw.point(xy = ((x2,y2)), fill = 'black')

        if mass[5] == 1000:
            x1 = x + (r1) * np.cos(angle)
            y1 = z + (r1) * np.sin(angle)
            x2 = x + (r1 + lenRay) * np.cos(angle)
            y2 = z + (r1 + lenRay) * np.sin(angle)
            x1 = 250 + x1 * multi
            x2 = 250 + x2 * multi
            y1 = 250 + y1 * multi
            y2 = 250 + y2 * multi
            draw.line(xy = ((x1,y1), (x2,y2)), fill = 'brown')
        else:
            x1 = x + (r1) * np.cos(angle)
            y1 = z + (r1) * np.sin(angle)
            x2 = x + (r1 + lenRay * mass[5] / 1000) * np.cos(angle)
            y2 = z + (r1 + lenRay * mass[5] / 1000) * np.sin(angle)
            x1 = 250 + x1 * multi
            x2 = 250 + x2 * multi
            y1 = 250 + y1 * multi
            y2 = 250 + y2 * multi
            draw.line(xy = ((x1,y1), (x2,y2)), fill = 'brown')
            draw.point(xy = ((x2,y2)), fill = 'black')

    def draw(self, x, z, mass, r1, r2, lenRay, angle, multi):
            self.drawEmptyRectangle(x, z, r1, r2, angle, multi)
            self.drawSensorLine(x, z, mass, r1, r2, lenRay, angle, multi)

    def restart(self, i):
        self.image.save("images\\test_{}.png".format(i), "PNG")
        self.image = Image.new("RGB", (500,500), (155,155,155))
