"""Build ``workbook_with_pivot.xlsx``, the fixture the workbook sink's preservation tests use.

Run it from the repo root:

    PYTHONPATH=src .venv/bin/python tests/fixtures/make_workbook_with_pivot.py

The fixture is committed, so this script is only needed when the fixture has to change. It is here
so the fixture is not an opaque blob nobody can rebuild.

Why it is built this way. No Python library writes a pivot table. ``openpyxl`` reads the pivot parts
of a workbook and writes back what it read, but it will not create one, and ``xlsxwriter`` has no
pivot support at all. So the base workbook is built with ``openpyxl`` -- which gets the sheets, the
styles, the content types and the relationships right -- and the four pivot parts are then added to
the zip by hand, with the three files that have to point at them patched.

The result is small on purpose: two sheets, four data rows, one pivot table, one cache. Small enough
to commit, and it carries the two things under test. One is the cached ``sharedItems`` values, which
are what fills a pivot filter dropdown before anyone refreshes it. The other is the ``extLst`` block
on the cache definition, which is where Excel keeps the parts of a pivot that postdate the original
file format -- slicer links among them.

The values are invented. There is no client data here.
"""

import zipfile
from pathlib import Path

import openpyxl

HERE = Path(__file__).parent
TARGET = HERE / "workbook_with_pivot.xlsx"

MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
DOC_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
CONTENT_TYPES = "http://schemas.openxmlformats.org/package/2006/content-types"

ROWS = [("A", 1), ("B", 2), ("A", 3), ("B", 4)]

PIVOT_CACHE_DEFINITION = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<pivotCacheDefinition xmlns="{MAIN}" xmlns:r="{DOC_REL}" r:id="rId1" refreshedBy="fixture"
 refreshedVersion="6" createdVersion="6" minRefreshableVersion="3" recordCount="4">
  <cacheSource type="worksheet"><worksheetSource ref="A1:B5" sheet="Data"/></cacheSource>
  <cacheFields count="2">
    <cacheField name="code" numFmtId="0">
      <sharedItems count="2"><s v="A"/><s v="B"/></sharedItems>
    </cacheField>
    <cacheField name="amount" numFmtId="0">
      <sharedItems containsSemiMixedTypes="0" containsString="0" containsNumber="1"
       containsInteger="1" minValue="1" maxValue="4" count="4">
        <n v="1"/><n v="2"/><n v="3"/><n v="4"/>
      </sharedItems>
    </cacheField>
  </cacheFields>
  <extLst>
    <ext uri="{{725AE2AE-9491-48be-B2B4-4EB974FC3084}}"
     xmlns:x14="http://schemas.microsoft.com/office/spreadsheetml/2009/9/main">
      <x14:pivotCacheDefinition slicerData="1" pivotCacheId="1234567"/>
    </ext>
  </extLst>
</pivotCacheDefinition>
"""

PIVOT_CACHE_RECORDS = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<pivotCacheRecords xmlns="{MAIN}" xmlns:r="{DOC_REL}" count="4">
  <r><x v="0"/><x v="0"/></r>
  <r><x v="1"/><x v="1"/></r>
  <r><x v="0"/><x v="2"/></r>
  <r><x v="1"/><x v="3"/></r>
</pivotCacheRecords>
"""

PIVOT_TABLE = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<pivotTableDefinition xmlns="{MAIN}" name="PivotTable1" cacheId="1" applyNumberFormats="0"
 applyBorderFormats="0" applyFontFormats="0" applyPatternFormats="0" applyAlignmentFormats="0"
 applyWidthHeightFormats="1" dataCaption="Values" updatedVersion="6" minRefreshableVersion="3"
 useAutoFormatting="1" itemPrintTitles="1" createdVersion="6" indent="0" outline="1"
 outlineData="1" multipleFieldFilters="0">
  <location ref="A3:B6" firstHeaderRow="1" firstDataRow="2" firstDataCol="1"/>
  <pivotFields count="2">
    <pivotField axis="axisRow" showAll="0">
      <items count="3"><item x="0"/><item x="1"/><item t="default"/></items>
    </pivotField>
    <pivotField dataField="1" showAll="0"/>
  </pivotFields>
  <rowFields count="1"><field x="0"/></rowFields>
  <rowItems count="3">
    <i><x/></i><i><x v="1"/></i><i t="grand"><x/></i>
  </rowItems>
  <colItems count="1"><i/></colItems>
  <dataFields count="1">
    <dataField name="Sum of amount" fld="1" baseField="0" baseItem="0"/>
  </dataFields>
  <pivotTableStyleInfo name="PivotStyleLight16" showRowHeaders="1" showColHeaders="1"
   showRowStripes="0" showColStripes="0" showLastColumn="1"/>
</pivotTableDefinition>
"""

CACHE_DEFINITION_RELS = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="{PKG_REL}">
  <Relationship Id="rId1"
   Type="{DOC_REL}/pivotCacheRecords" Target="pivotCacheRecords1.xml"/>
</Relationships>
"""

PIVOT_TABLE_RELS = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="{PKG_REL}">
  <Relationship Id="rId1"
   Type="{DOC_REL}/pivotCacheDefinition" Target="../pivotCache/pivotCacheDefinition1.xml"/>
</Relationships>
"""


def _base_workbook(path: Path) -> None:
    """Two sheets: the rows the pivot summarises, and the sheet the pivot sits on."""
    book = openpyxl.Workbook()
    book.remove(book.active)

    data = book.create_sheet("Data")
    data["A1"] = "code"
    data["B1"] = "amount"
    for index, (code, amount) in enumerate(ROWS, start=2):
        data[f"A{index}"] = code
        data[f"B{index}"] = amount

    pivot = book.create_sheet("Pivot")
    pivot["A1"] = "Built by hand; the flow never names this tab."

    book.save(path)
    book.close()


def _patched(name: str, text: str) -> str:
    """The three parts that have to learn about the pivot, each patched with one insertion."""
    if name == "[Content_Types].xml":
        additions = (
            '<Override PartName="/xl/pivotCache/pivotCacheDefinition1.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.pivotCacheDefinition+xml"/>'
            '<Override PartName="/xl/pivotCache/pivotCacheRecords1.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.pivotCacheRecords+xml"/>'
            '<Override PartName="/xl/pivotTables/pivotTable1.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.pivotTable+xml"/>'
        )
        return text.replace("</Types>", additions + "</Types>")

    if name == "xl/_rels/workbook.xml.rels":
        addition = (
            f'<Relationship Id="rIdPivotCache1" '
            f'Type="{DOC_REL}/pivotCacheDefinition" Target="pivotCache/pivotCacheDefinition1.xml"/>'
        )
        return text.replace("</Relationships>", addition + "</Relationships>")

    if name == "xl/workbook.xml":
        addition = '<pivotCaches><pivotCache cacheId="1" r:id="rIdPivotCache1"/></pivotCaches>'
        return text.replace("</workbook>", addition + "</workbook>")

    return text


def build(target: Path = TARGET) -> Path:
    scratch = target.with_suffix(".base.xlsx")
    _base_workbook(scratch)

    with zipfile.ZipFile(scratch) as source, zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as out:
        for item in source.infolist():
            out.writestr(item.filename, _patched(item.filename, source.read(item.filename).decode("utf-8")))

        out.writestr("xl/pivotCache/pivotCacheDefinition1.xml", PIVOT_CACHE_DEFINITION)
        out.writestr("xl/pivotCache/pivotCacheRecords1.xml", PIVOT_CACHE_RECORDS)
        out.writestr("xl/pivotCache/_rels/pivotCacheDefinition1.xml.rels", CACHE_DEFINITION_RELS)
        out.writestr("xl/pivotTables/pivotTable1.xml", PIVOT_TABLE)
        out.writestr("xl/pivotTables/_rels/pivotTable1.xml.rels", PIVOT_TABLE_RELS)
        out.writestr(
            "xl/worksheets/_rels/sheet2.xml.rels",
            f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            f'<Relationships xmlns="{PKG_REL}">'
            f'<Relationship Id="rIdPivot1" Type="{DOC_REL}/pivotTable" Target="../pivotTables/pivotTable1.xml"/>'
            f"</Relationships>\n",
        )

    scratch.unlink()
    return target


if __name__ == "__main__":
    built = build()
    # Prove it loads, so a broken fixture is found here rather than in a test failure.
    book = openpyxl.load_workbook(built)
    pivots = sum(len(book[tab]._pivots) for tab in book.sheetnames)
    print(f"{built} -- {built.stat().st_size} bytes, tabs {book.sheetnames}, pivot tables {pivots}")
    book.close()
