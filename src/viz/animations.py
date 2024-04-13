from manim import *
import numpy as np

class GraphZoomScene(MovingCameraScene):

    def construct(self):
        # Set the background color
        self.camera.background_color = BLACK
        
        # Create a graph
        graph = Graph(
            [0, 1, 2, 3],
            [(0,1), (1,2), (2,3), (3,0)],
            layout={0: [-1,0,0], 1: [0,1,0], 2: [1,0,0], 3: [0,-1,0]},
            layout_scale=5,
            labels=False,
            vertex_config={"radius": 0.2},
        )

        frame = self.camera.frame
        frame.save_state()
        
        # Add graph to the scene
        self.play(Create(graph))
        self.wait(1)


        
        ## Focus on node 0
        node_focus = graph[0]
        
        self.play(
            frame.animate.set(width=node_focus.width * 5).move_to(node_focus.get_center())
        )
        self.wait(2)
#
        #self.play(
        #    self.camera.frame.animate.set(width=node_focus.width / 5)
        #)
        #self.play(
        #    self.camera.frame.animate.move_to(graph.get_center())
        #)
#
        ## Create a time series within the zoomed area
        #time_series_data = self.get_time_series_data()
        #time_series = self.create_time_series(time_series_data)
#
        #time_series_window = AnimationGroup(
        #    GrowFromPoint(time_series, node_focus.get_center()),
        #    self.camera.frame.animate.move_to(time_series.get_center())
        #, run_time = 5)
        #self.play(time_series_window)
        #
        
        #self.play(Write(time_series))
        #self.wait(2)
        #
        ## Create a time series within the zoomed area
        #time_series_data = self.get_time_series_data()
        #time_series = self.create_time_series(time_series_data)
        #
        #self.play(FadeIn(node_focus))

        #
        # Transition to the new scene
        #self.play(FadeOut(node_focus))

    def get_time_series_data(self):
        # Generate random time series data
        data = np.random.rand(10)
        return data

    def create_time_series(self, data):
        # Convert data into a line chart
        time_series = VGroup()
        time_series.set_fill(color=GRAY_A, opacity=0.8)
        time_series.set_stroke(width=1, color=WHITE)

        for i in range(1, len(data)):
            start_point = np.array([i-1, data[i-1], 0])
            end_point = np.array([i, data[i], 0])
            line = Line(start_point, end_point, stroke_width=2, color=BLUE)
            time_series.add(line)
        
        return time_series


if __name__ == "__main__":
    graph_zoom_scene = GraphZoomScene()
    graph_zoom_scene.render(preview=True)

# Run this with:
# manim -p -ql --resolution 1920,1080 your_script_name.py GraphZoomScene



class TrafficNodeMovement(Scene):
    def construct(self):
        fig = plt.figure(figsize=(10, 8))
        ax = plt.gca()

        def update(i):
            ax.clear()
            ax.set_title(f'Time Traffic at Step {i}')
            ax.set_xlabel('Node')
            ax.set_ylabel('Traffic Volume')
            ax.set_ylim(0, np.max(raw_data_features[:, :, 0]))

            nodes = distance_matrix[['from', 'to']].values.tolist()
            nodes = [[*node, {"cost":cost}] for node, cost in zip(nodes, distance_matrix['cost'])]
            G = nx.Graph()
            G.add_edges_from(nodes)

            nx.draw(G, 
                    with_labels=False, 
                    node_size=raw_data_features[i, :, 0],
                    node_color=raw_data_features[i, :, 0],
                    ax=ax)

        ani = animation.FuncAnimation(fig, update, frames=raw_data_features.shape[0], interval=200)
        self.add(ani.to_mobject())