import asyncio, traceback

async def test():
    try:
        from app.graph.nova_graph import run_graph
        result = await run_graph('tesla news', {'news': True, 'summary': True, 'sentiment': True})
        print('Keys:', list(result.keys()))
        print('Success:', result.get('success'))
        print('Errors:', result.get('errors'))
        print('News count:', len(result.get('data', {}).get('news', [])))
    except Exception as e:
        print('EXCEPTION:', e)
        traceback.print_exc()

asyncio.run(test())
