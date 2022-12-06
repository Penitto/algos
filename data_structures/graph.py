from collections import defaultdict


class Graph:

    def __init__(self):

        self.graph = defaultdict(list)

    def addEdge(self, u, v):

        self.graph[u].append(v)

    def BFS(self, start):

        visited = defaultdict(bool)

        queue = []

        queue.append(start)
        visited[start] = True

        while queue:
            start = queue.pop(0)

            # print(start)
            for i in self.graph[start]:
                if not visited[i]:
                    queue.append(i)
                    visited[i] = True

    def DFSUtil(self, v, visited):

        # Mark the current node as visited and print it
        visited[v] = True
        print(v)

        # Recur for all the vertices adjacent to
        # this vertex
        for i in self.graph[v]:
            if not visited[i]:
                self.DFSUtil(i, visited)

    # The function to do DFS traversal. It uses
    # recursive DFSUtil()
    def DFS(self):
        # Mark all the vertices as not visited
        visited = defaultdict(bool)

        # Call the recursive helper function to print
        # DFS traversal starting from all vertices one
        # by one
        for i in range(len(self.graph)):
            if not visited[i]:
                self.DFSUtil(i, visited)
