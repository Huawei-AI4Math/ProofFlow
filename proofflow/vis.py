import json
import os
import webbrowser

import matplotlib.pyplot as plt
import networkx as nx


def _get_item_value(item, key, default=None):
    """
    Helper function to get values from either Pydantic models or dictionaries.
    Works with both item.key and item[key] access patterns.
    """
    try:
        # Try Pydantic model attribute access first
        if hasattr(item, key):
            return getattr(item, key)
        # Fall back to dictionary access
        elif isinstance(item, dict):
            return item.get(key, default)
        else:
            return default
    except (AttributeError, KeyError):
        return default


def build_dag(data):
    """Build a directed acyclic graph from the data structure.
    
    Args:
        data: List of items, each can be either a Pydantic model or dictionary
    """
    G = nx.DiGraph()
    
    # Create a dictionary to store node information
    node_info = {}
    
    for item in data:
        node_id = _get_item_value(item, 'id')

        if not node_id:
            continue  # Skip items without ID
        
        # Determine the type based on ID prefix
        if node_id.startswith('tc_'):
            item_type = 'condition'
        elif node_id.startswith('ts_'):
            item_type = 'solution'
        elif node_id.startswith('l'):
            item_type = 'lemma'
        elif node_id.startswith('def'):
            item_type = 'definition'
        else:
            item_type = 'unknown'
        
        # Convert Pydantic model to dict if needed, or copy existing dict
        if hasattr(item, 'model_dump'):
            item_dict = item.model_dump()
        else:
            item_dict = item.copy() if isinstance(item, dict) else dict(item)
        
        # Add type field to the item
        item_dict['type'] = item_type
        
        node_info[node_id] = item_dict

        # Check if there is a field called formalization and clean up attempt history
        # Remove attempt_history from formalization to keep graph data clean
        if ('formalization' in item_dict and
                isinstance(item_dict['formalization'], dict)):
            item_dict['formalization'].pop('attempt_history', None)
        
        # Check if there is a field called solved_lemma and clean up attempt history
        # Remove attempt_history from solved_lemma to avoid storing unnecessary data
        if ('solved_lemma' in item_dict and
                isinstance(item_dict['solved_lemma'], dict)):
            item_dict['solved_lemma'].pop('attempt_history', None)

        # Add node to graph
        G.add_node(node_id)
        
        # Add edges based on dependencies
        dependencies = _get_item_value(item, 'dependencies', [])
        for dep in dependencies:
            if dep:  # Only add edge if dependency exists
                G.add_edge(dep, node_id)
    
    return G, node_info


def create_static_visualization(G, node_info, filename='proof_graph.png', dpi: int = 100):
    """Create a static PNG visualization of the proof graph."""
    plt.figure(figsize=(8, 6))
    
    # Use built-in NetworkX layout - no external dependencies needed
    nx.spring_layout(G, k=3, iterations=50, seed=42)
    
    # Color nodes based on type
    node_colors = []
    for node in G.nodes():
        node_type = node_info[node].get('type', 'unknown')
        if node_type == 'condition':
            node_colors.append('#ffcccc')  # Light red for conditions
        elif node_type == 'solution':
            node_colors.append('#ccffcc')  # Light green for solutions
        elif node_type == 'definition':
            node_colors.append("#c0a07f")  # Light orange for definitions
        else:  # lemma
            node_colors.append('#ccccff')  # Light blue for lemmas
    
    # Draw the graph
    nx.draw_spring(G,
                   node_color=node_colors,
                   node_size=2000,
                   font_size=12,
                   font_weight='bold',
                   with_labels=True,
                   arrows=True,
                   arrowsize=20,
                   edge_color='gray',
                   linewidths=2,
                   node_shape='o')
    
    # Add legend
    legend_elements = [
        plt.scatter([], [], c='#ffcccc', s=200,
                    label='Theorem Conditions (tc)'),
        plt.scatter([], [], c='#ccccff', s=200, label='Lemmas (l)'),
        plt.scatter([], [], c='#ccffcc', s=200,
                    label='Theorem Solutions (ts)'),
        plt.scatter([], [], c='#c0a07f', s=200, label='Definitions (def)')
    ]
    plt.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    plt.title('Proof Dependency Graph\n(Arrow direction shows dependency flow)',
              fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    print(f"Static visualization saved as {filename}")
    plt.show()


def create_interactive_visualization(G, node_info, proof_str="", filename='proof_graph_interactive.html'):
    """Create an interactive HTML visualization using pyvis with split screen layout."""
    
    # Process proof_str to handle escape sequences
    if proof_str:
        proof_str = proof_str.replace('\\t', '\t').replace('\\n', '\n')
    
    # Build nodes and edges data for vis.js
    nodes_data = []
    edges_data = []
    
    # Add nodes
    for node in G.nodes():
        info = node_info[node]
        node_type = info.get('type', 'unknown')
        
        # Determine contour color based on lean verification status
        contour_color = '#fa2c07'  # Default grey
        
        if node_type in ['condition', 'definition']:
            # For theorem conditions and definitions, only check formalization.lean_pass
            if info.get('formalization', {}).get('lean_pass', False):
                contour_color = '#00ff00'  # Green
        else:
            # For lemmas and solutions, check solved_lemma.lean_verify first,
            # then formalization.lean_pass
            if info.get('solved_lemma', {}).get('lean_verify', False):
                contour_color = '#00ff00'  # Strong green
            elif info.get('formalization', {}).get('lean_pass', False):
                contour_color = '#FFA500'  # Orange
        
        # Set node properties based on type
        if node_type == 'condition':
            color = '#eba0a0'
            shape = 'box'
            size = 30
        elif node_type == 'solution':
            color = '#a3c2a8'
            shape = 'star'
            size = 40
        elif node_type == 'definition':
            color = "#cfb795"
            shape = 'box'
            size = 30
        else:  # lemma or unknown
            color = '#8dafcc'
            shape = 'dot'
            size = 25
        
        # Extract label from node ID for better display
        if node.startswith('l'):
            label = node  # Keep l1, l2, etc.
        elif node.startswith('ts_'):
            label = node  # Keep ts_1, ts_2, etc.
        else:
            label = node
            
        nodes_data.append({
            'id': node,
            'label': label,
            'color': {
                'background': color,
                'border': contour_color,
                'highlight': {
                    'background': color,
                    'border': contour_color
                }
            },
            'shape': shape,
            'size': size,
            'font': {'size': 14, 'color': '#000000'},
            'labelHighlightBold': False,
            'borderWidth': 3,
            'chosen': False,
            'labelHighlightBold': False
        })
    
    # Add edges
    for edge in G.edges():
        edges_data.append({
            'from': edge[0],
            'to': edge[1],
            'arrows': 'to',
            'color': {
                'color': '#666666',
                'highlight': '#666666',
                'hover': '#666666'
            },
            'width': 2
        })
    
    # Create custom HTML with split screen layout
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <title>Proof Graph Visualization</title>
        <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"></script>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.css" rel="stylesheet" type="text/css" />
        <style>
            body {{
                margin: 0;
                padding: 0;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                height: 100vh;
                overflow: hidden;
            }}
            
            .container {{
                display: flex;
                height: 100vh;
                background: #f0f0f0;
            }}
            
            #left-panel {{
                flex: 1;
                display: flex;
                flex-direction: column;
                background: white;
                border-right: 2px solid #ddd;
                height: 100vh;
                overflow: hidden;
            }}
            
            #proof-str-panel {{
                background: #f8f9fa;
                border-bottom: 2px solid #ddd;
                padding: 15px;
                height: 200px;
                overflow-y: auto;
                flex-shrink: 0;
            }}
            
            #proof-str-title {{
                font-weight: 600;
                color: #2d3748;
                margin-bottom: 10px;
                font-size: 14px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            
            #proof-str-content {{
                color: #4a5568;
                line-height: 1.6;
                white-space: pre-wrap;
                word-wrap: break-word;
                font-family: 'Monaco', 'Courier New', monospace;
                font-size: 13px;
                background: white;
                padding: 12px;
                border-radius: 4px;
                border: 1px solid #e2e8f0;
            }}
            
            #graph-container {{
                flex: 1;
                position: relative;
                background: white;
                min-height: 0;
                overflow: hidden;
            }}
            
            #mynetwork {{
                width: 100%;
                height: 100%;
            }}
            
            #info-panel {{
                width: 40%;
                max-width: 600px;
                background: white;
                overflow-y: auto;
                padding: 20px;
                box-shadow: -2px 0 10px rgba(0,0,0,0.1);
            }}
            
            #info-panel.hidden {{
                display: none;
            }}
            
            .info-header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 20px;
            }}
            
            .info-header h2 {{
                margin: 0;
                font-size: 24px;
            }}
            
            .info-header .node-type {{
                opacity: 0.9;
                font-size: 14px;
                margin-top: 5px;
            }}
            
            .field-group {{
                background: #f8f9fa;
                border-left: 4px solid #667eea;
                padding: 15px;
                margin-bottom: 15px;
                border-radius: 4px;
            }}
            
            .field-name {{
                font-weight: 600;
                color: #2d3748;
                margin-bottom: 8px;
                font-size: 14px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            
            .field-value {{
                color: #4a5568;
                line-height: 1.6;
                white-space: pre-wrap;
                word-wrap: break-word;
                font-family: 'Monaco', 'Courier New', monospace;
                font-size: 13px;
            }}
            
            .field-value.code {{
                color: #4a5568;
                padding: 12px;
                border-radius: 4px;
                overflow-x: auto;
            }}
            
            .sub-dict {{
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 4px;
                padding: 10px;
                margin: 8px 0;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }}
            
            .sub-dict-title {{
                font-weight: 600;
                color: #4a5568;
                margin-bottom: 8px;
                font-size: 12px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                border-bottom: 1px solid #e2e8f0;
                padding-bottom: 4px;
            }}
            
            .sub-dict-content {{
                color: #2d3748;
                line-height: 1.5;
                font-size: 12px;
            }}
            
            .placeholder {{
                text-align: center;
                color: #718096;
                padding: 40px;
                font-size: 18px;
            }}
            
            .placeholder-icon {{
                font-size: 48px;
                margin-bottom: 20px;
                opacity: 0.3;
            }}
            
            #toggle-panel {{
                position: absolute;
                right: 10px;
                top: 10px;
                z-index: 1000;
                background: #667eea;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 14px;
                transition: background 0.3s;
            }}
            
            #toggle-panel:hover {{
                background: #764ba2;
            }}
            
            .legend {{
                position: absolute;
                top: 20px;
                left: 20px;
                background: white;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                z-index: 100;
                max-width: 300px;
            }}
            
            .legend-section {{
                margin-bottom: 15px;
            }}
            
            .legend-section:last-child {{
                margin-bottom: 0;
            }}
            
            .legend-title {{
                font-weight: 600;
                font-size: 12px;
                color: #2d3748;
                margin-bottom: 8px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                border-bottom: 1px solid #e2e8f0;
                padding-bottom: 4px;
            }}
            
            .legend-item {{
                display: flex;
                align-items: center;
                margin-bottom: 8px;
            }}
            
            .legend-item:last-child {{
                margin-bottom: 0;
            }}
            
            .legend-color {{
                width: 20px;
                height: 20px;
                margin-right: 10px;
                border-radius: 3px;
            }}
            
            .resizer {{
                background: #ddd;
                cursor: col-resize;
                width: 4px;
                height: 100%;
                position: absolute;
                right: 0;
                top: 0;
                z-index: 1000;
            }}
            
            .resizer:hover {{
                background: #bbb;
            }}
            
            .resizer-vertical {{
                background: #ddd;
                cursor: row-resize;
                height: 4px;
                width: 100%;
                position: absolute;
                bottom: 0;
                left: 0;
                z-index: 1000;
            }}
            
            .resizer-vertical:hover {{
                background: #bbb;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div id="left-panel">
                <div id="proof-str-panel">
                    <div id="proof-str-title">Informal theorem and proof</div>
                    <div id="proof-str-content">{proof_str}</div>
                    <div class="resizer-vertical" id="vertical-resizer"></div>
                </div>
                <div id="graph-container">
                    <button id="toggle-panel" onclick="togglePanel()">Toggle Info Panel</button>
                    <div class="legend">
                        <div class="legend-section">
                            <div class="legend-title">Node Types</div>
                            <div class="legend-item">
                                <div class="legend-color" style="background: #eba0a0;"></div>
                                <span>Theorem Conditions (tc)</span>
                            </div>
                            <div class="legend-item">
                                <div class="legend-color" style="background: #cfb795 ;"></div>
                                <span>Theorem Definitions (def)</span>
                            </div>
                            <div class="legend-item">
                                <div class="legend-color" style="background: #8dafcc;"></div>
                                <span>Lemmas (l)</span>
                            </div>
                            <div class="legend-item">
                                <div class="legend-color" style="background: #a3c2a8;"></div>
                                <span>Theorem Solutions (ts)</span>
                            </div>

                        </div>
                        <div class="legend-section">
                            <div class="legend-title">Verification Status</div>
                            <div class="legend-item">
                                <div class="legend-color" style="background: #FFFFFF; border: 2px solid #fa2c07;"></div>
                                <span>Formalization failed</span>
                            </div>
                            <div class="legend-item">
                                <div class="legend-color" style="background: #FFFFFF; border: 2px solid #FFA500;"></div>
                                <span>Formalization passed</span>
                            </div>
                            <div class="legend-item">
                                <div class="legend-color" style="background: #FFFFFF; border: 2px solid #00ff00;"></div>
                                <span>Fully verified</span>
                            </div>
                        </div>
                    </div>
                    <div id="mynetwork"></div>
                </div>
            </div>
            
            <div class="resizer" id="horizontal-resizer"></div>
            
            <div id="info-panel">
                <div id="info-content">
                    <div class="placeholder">
                        <div class="placeholder-icon">📊</div>
                        <p>Click on a node to view its details</p>
                    </div>
                </div>
            </div>
        </div>
        
        <script type="text/javascript">
            // Store node information
            var nodeInfo = {json.dumps(node_info)};
            
            // Create nodes and edges
            var nodes = new vis.DataSet({json.dumps(nodes_data)});
            var edges = new vis.DataSet({json.dumps(edges_data)});
            
            // Create network
            var container = document.getElementById('mynetwork');
            var data = {{
                nodes: nodes,
                edges: edges
            }};
            
            var options = {{
                physics: {{
                    enabled: true,
                    solver: 'hierarchicalRepulsion',
                    hierarchicalRepulsion: {{
                        centralGravity: 0.0,
                        springLength: 200,
                        springConstant: 0.01,
                        nodeDistance: 150,
                        damping: 0.09
                    }}
                }},
                edges: {{
                    smooth: {{
                        type: 'continuous',
                        forceDirection: 'none'
                    }},
                    color: {{
                        color: '#666666',
                        highlight: '#666666',
                        hover: '#666666'
                    }},
                    width: 2
                }},
                interaction: {{
                    hover: true,
                    tooltipDelay: 100,
                    navigationButtons: true,
                    keyboard: true
                }},
                nodes: {{
                    font: {{
                        size: 14,
                        color: '#000000'
                    }},
                    labelHighlightBold: false,
                    chosen: false,
                    shapeProperties: {{
                        useBorderWithImage: false
                    }},
                    scaling: {{
                        label: {{
                            enabled: true,
                            min: 8,
                            max: 20,
                            maxVisible: 20,
                            drawThreshold: 5
                        }}
                    }}
                }}
            }};
            
            var network = new vis.Network(container, data, options);
            
            // Format field value for display
            function formatFieldValue(value, fieldName) {{
                if (value === null || value === undefined) {{
                    return 'N/A';
                }}
                
                // Handle objects (dictionaries) as sub-boxes
                if (typeof value === 'object' && !Array.isArray(value)) {{
                    var html = '';
                    for (var key in value) {{
                        if (value.hasOwnProperty(key)) {{
                            var subValue = value[key];
                            var subValueStr = '';
                            
                            if (typeof subValue === 'object' &&
                                    !Array.isArray(subValue)) {{
                                // Nested object - show as JSON
                                subValueStr = JSON.stringify(subValue, null, 2);
                            }} else if (Array.isArray(subValue)) {{
                                // Array - show as JSON
                                subValueStr = JSON.stringify(subValue, null, 2);
                            }} else {{
                                // Primitive value
                                subValueStr = String(subValue);
                            }}
                            
                            // Replace escaped newlines and tabs with actual characters
                            subValueStr = subValueStr.replace(/\\\\n/g, '\\n');
                            subValueStr = subValueStr.replace(/\\\\t/g, '\\t');
                            subValueStr = subValueStr.replace(/\\t/g, '\\t');
                            
                            html += '<div class="sub-dict">';
                            html += '<div class="sub-dict-title">' + key + '</div>';
                            html += '<div class="sub-dict-content">' + subValueStr + '</div>';
                            html += '</div>';
                        }}
                    }}
                    return html;
                }}
                
                // Convert to string for primitive values and arrays
                var strValue = String(value);
                
                // Handle arrays
                if (Array.isArray(value)) {{
                    strValue = JSON.stringify(value, null, 2);
                }}
                
                // Replace escaped newlines and tabs with actual characters
                strValue = strValue.replace(/\\\\n/g, '\\n');
                strValue = strValue.replace(/\\\\t/g, '\\t');
                strValue = strValue.replace(/\\t/g, '\\t');
                
                return strValue;
            }}
            
            // Handle node click events
            network.on("click", function(params) {{
                if (params.nodes.length > 0) {{
                    var nodeId = params.nodes[0];
                    var info = nodeInfo[nodeId];
                    
                    if (info) {{
                        var html = '<div class="info-header">';
                        html += '<h2>' + nodeId + '</h2>';
                        var nodeType = info.type || 'unknown';
                        var capitalizedType = nodeType.charAt(0).toUpperCase() +
                            nodeType.slice(1);
                        html += '<div class="node-type">Type: ' + capitalizedType +
                            '</div>';
                        html += '</div>';
                        
                        // Add all fields
                        for (var key in info) {{
                            if (key === 'type' || key === 'id') continue;
                            
                            var fieldName = key.replace(/_/g, ' ')
                                .replace(/\\b\\w/g, function(l) {{
                                    return l.toUpperCase();
                                }});
                            var fieldValue = formatFieldValue(info[key], fieldName);
                            
                            if (fieldValue && fieldValue !== 'N/A') {{
                                // Special formatting for code fields
                                var isCode = key.includes('code') ||
                                    key.includes('lean') ||
                                    key.includes('statement');
                                
                                html += '<div class="field-group">';
                                html += '<div class="field-name">' + fieldName +
                                    '</div>';
                                html += '<div class="field-value' +
                                    (isCode ? ' code' : '') + '">' +
                                    fieldValue + '</div>';
                                html += '</div>';
                            }}
                        }}
                        
                        document.getElementById('info-content')
                            .innerHTML = html;
                    }}
                }}
            }});
            
            // Toggle panel function
            function togglePanel() {{
                var panel = document.getElementById('info-panel');
                panel.classList.toggle('hidden');
                
                // Resize network when panel is toggled
                setTimeout(function() {{
                    network.fit();
                }}, 300);
            }}
            
            // Initial network fit
            network.once('stabilized', function() {{
                network.fit();
            }});
            
            // Resizable panels functionality
            function makeResizable() {{
                const verticalResizer = document.getElementById('vertical-resizer');
                const horizontalResizer = document.getElementById('horizontal-resizer');
                const proofPanel = document.getElementById('proof-str-panel');
                const graphContainer = document.getElementById('graph-container');
                const infoPanel = document.getElementById('info-panel');
                const leftPanel = document.getElementById('left-panel');
                
                let isVerticalResizing = false;
                let isHorizontalResizing = false;
                
                // Vertical resizer (between proof panel and graph)
                verticalResizer.addEventListener('mousedown', function(e) {{
                    isVerticalResizing = true;
                    document.addEventListener('mousemove', verticalResize);
                    document.addEventListener('mouseup', stopVerticalResize);
                    e.preventDefault();
                }});
                
                function verticalResize(e) {{
                    if (!isVerticalResizing) return;
                    
                    const containerHeight = leftPanel.offsetHeight;
                    const newHeight = e.clientY - leftPanel.offsetTop;
                    const minHeight = 100;
                    const maxHeight = containerHeight - 100;
                    
                    if (newHeight >= minHeight && newHeight <= maxHeight) {{
                        proofPanel.style.height = newHeight + 'px';
                        graphContainer.style.height = (containerHeight - newHeight) + 'px';
                        network.fit();
                    }}
                }}
                
                function stopVerticalResize() {{
                    isVerticalResizing = false;
                    document.removeEventListener('mousemove', verticalResize);
                    document.removeEventListener('mouseup', stopVerticalResize);
                }}
                
                // Horizontal resizer (between left and right panels)
                horizontalResizer.addEventListener('mousedown', function(e) {{
                    isHorizontalResizing = true;
                    document.addEventListener('mousemove', horizontalResize);
                    document.addEventListener('mouseup', stopHorizontalResize);
                    e.preventDefault();
                }});
                
                function horizontalResize(e) {{
                    if (!isHorizontalResizing) return;
                    
                    const containerWidth = document.querySelector('.container').offsetWidth;
                    const newWidth = e.clientX - leftPanel.offsetLeft;
                    const minWidth = 300;
                    const maxWidth = containerWidth - 300;
                    
                    if (newWidth >= minWidth && newWidth <= maxWidth) {{
                        leftPanel.style.flex = 'none';
                        leftPanel.style.width = newWidth + 'px';
                        infoPanel.style.width = (containerWidth - newWidth) + 'px';
                        network.fit();
                    }}
                }}
                
                function stopHorizontalResize() {{
                    isHorizontalResizing = false;
                    document.removeEventListener('mousemove', horizontalResize);
                    document.removeEventListener('mouseup', stopHorizontalResize);
                }}
            }}
            
            // Initialize resizable panels
            makeResizable();
        </script>
    </body>
    </html>
    """
    
    # Save the HTML file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Interactive visualization saved as {filename}")
    
    # Open in browser automatically
    webbrowser.open('file://' + os.path.realpath(filename))
    
    return None