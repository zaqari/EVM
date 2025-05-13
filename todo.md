# Required tasks

### [1] Update Recurrence Plot in EDA
1. [ ] Create square matrix from columnar dataset
2. [ ] change heatmap to be plotly express
3. [ ] Use following to update the template to have custom hover labels

```python
fig.update_traces(
    hovertemplate="<b>X: %{x}</b><br>"+
                  "<b>Y: %{y}</b><br>"+
                  "<b>Z: %{z}</b><br>"+
                  "<b>Info: %{customdata}</b><extra></extra>",
    customdata=custom_data #an array of custom data text
)

fig.show()
```