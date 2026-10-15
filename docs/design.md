# Pipeline design

The [editable Figma pipeline](https://www.figma.com/design/bRUVlmkQKApYB2bWdHiblv?node-id=29-2) combines document indexing, retrieval and answer checks in one frame. It uses the project's default settings.

| File | Purpose |
| --- | --- |
| [Pipeline SVG](../assets/figures/pipeline.svg) | README diagram; can also be imported into Figma. |
| [Figma layout snapshot](../assets/figma/pipeline.json) | Frame ID, dimensions, editable text, shapes and connector coordinates. |
| [Packaged SVG](../src/scinexusrag/web/images/pipeline.svg) | The same diagram served by the app. |

Keep both SVG files identical and update the layout snapshot when the Figma frame changes. Check labels against the [pipeline implementation guide](pipeline.md).

The earlier [overview](https://www.figma.com/design/bRUVlmkQKApYB2bWdHiblv?node-id=8-2), [retrieval](https://www.figma.com/design/bRUVlmkQKApYB2bWdHiblv?node-id=12-2) and [answer-check](https://www.figma.com/design/bRUVlmkQKApYB2bWdHiblv?node-id=13-20) frames remain in the same Figma file. Detailed retrieval and answer-check diagrams remain in the implementation guide.
