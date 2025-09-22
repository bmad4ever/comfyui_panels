# ComfyUI Panels

This packages contains nodes for generating and-or organizing comics-manga like panels, similarly to the ones below (_images contain the workflow_).

<img src="./workflows/Barebones Board Generation.png" alt="drawing" style="width:250px;"/>
<img src="./workflows/Character Over Panels.png" alt="drawing" style="width:250px;"/>


## Panel Layouts

Panel Layouts are an abstract hierarchical representation of a series of cuts made on a page in order to create the comics-manga panels. 

Starting with an empty page (a single full-page panel), a single cut, or multiple equally spaced cuts, divide this page into distinct panels.
Following cuts are made to existing panels, and these cuts can be nested to an arbitrary level of complexity.

Cuts are defined by:
- direction — horizontal or vertical.
- position — from a selection of 5 potential relative positions to the panel being cut.
- angle — a slight deviation from the defined direction for aesthetic purposes.
-  number of cuts — only applicable for one of the available relative positions, where the cuts will be forced to have the same spacing between each other.

Some layouts are provided in the [sample_layouts](sample_layouts) folder, but you can create your own and store them using the *Save Panel Layout* node. 

The image below is from one of the sample layouts.

<img src="./sample_layouts/PanelLayout (1).png" alt="drawing" style="width:200px;"/>

A Panel Layout does **NOT** store any particularaties regarding its panels, such as: corner shape, margins relative to the page's borders, line thickness or lack thereof, etc. 
It can, however, implicitly define relative spacing between panels via the order of the cuts, but not the concrete values.

The order of the layout's panels, and their relative spacing, is defined by the order of the cuts. 
The first cut(s) will have a bigger margin, suggesting a reading direction parallel to the cut for the nested panels. 
**The panel order is defined with this suggestion in mind**;
to deviate from this pattern you need to change the panels, or their order, separately from the layout. 

When saving, previewing, or processing, a layout you have the option to do it from left-to-right or from right-to-left.
These options do not change the underlying data — you can use a layout saved as left-to-right as a right-to-left layout — the option only exists to help you keep things visually coherent with respect to your reading direction. 



## Panel Layouts   

